"""Run three deterministic agent workflows using the installed package.

Run from the repository: python examples/run_all.py
"""

import asyncio

from agent_regression import run_suite
from agent_regression.demo import (
    crm_scenarios,
    make_crm_agent,
    make_search_agent,
    make_support_agent,
    search_scenarios,
    support_scenarios,
)
from agent_regression.reporting import render_terminal


async def main() -> None:
    for factory, scenarios in [
        (make_support_agent, support_scenarios()),
        (make_search_agent, search_scenarios()),
        (make_crm_agent, crm_scenarios()),
    ]:
        suite = await run_suite(factory, scenarios, repeats=3)
        render_terminal(suite)
        assert suite.failed_runs == 0


if __name__ == "__main__":
    asyncio.run(main())
