from __future__ import annotations

from agent_regression.schema import RequiredCall, Scenario, ToolCall
from agent_regression.suggest import suggest_checks


def test_suggestions_are_observations_in_first_seen_order_and_do_not_approve() -> None:
    scenario = Scenario(
        name="possibly_bad_recording",
        prompt="Refund an ineligible order",
        recorded_calls=[
            ToolCall(tool_name="lookup", arguments={"id": "123"}),
            ToolCall(tool_name="refund", arguments={"id": "123"}),
            ToolCall(tool_name="lookup", arguments={"id": "123"}, outcome="retry"),
        ],
    )
    before = scenario.model_dump()
    suggestions = suggest_checks(scenario)
    assert suggestions == [
        RequiredCall(tool_name="lookup", arguments={"id": "123"}, min_calls=2),
        RequiredCall(tool_name="refund", arguments={"id": "123"}),
    ]
    assert scenario.model_dump() == before
    assert not scenario.reviewed
    assert all(isinstance(check, RequiredCall) for check in suggestions)


def test_suggestions_preserve_boolean_numeric_distinction_and_ignore_argument_key_order() -> None:
    scenario = Scenario(
        name="test",
        prompt="request",
        recorded_calls=[
            ToolCall(tool_name="tool", arguments={"a": False, "b": 1}),
            ToolCall(tool_name="tool", arguments={"b": 1, "a": False}),
            ToolCall(tool_name="tool", arguments={"a": 0, "b": 1}),
        ],
    )
    suggestions = suggest_checks(scenario)
    assert len(suggestions) == 2
    assert isinstance(suggestions[0], RequiredCall)
    assert suggestions[0].min_calls == 2


def test_suggested_argument_containers_do_not_alias_original_recording() -> None:
    scenario = Scenario(
        name="test",
        prompt="request",
        recorded_calls=[ToolCall(tool_name="tool", arguments={"nested": {"id": "123"}})],
    )
    suggestion = suggest_checks(scenario)[0]
    assert isinstance(suggestion, RequiredCall)
    assert suggestion.arguments is not None
    nested = suggestion.arguments["nested"]
    assert isinstance(nested, dict)
    nested["id"] = "changed nested container"
    assert scenario.recorded_calls[0].arguments == {"nested": {"id": "123"}}
    suggestion.arguments["nested"] = {"id": "456"}
    assert scenario.recorded_calls[0].arguments == {"nested": {"id": "123"}}


def test_no_recorded_calls_has_no_invented_checks() -> None:
    scenario = Scenario(name="test", prompt="request")
    assert suggest_checks(scenario) == []
