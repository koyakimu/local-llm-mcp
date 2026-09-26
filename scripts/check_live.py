# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp>=2.2,<3"]
# ///
"""End-to-end check against a running local LLM: starts the server over stdio like an MCP client would.

Usage: uv run scripts/check_live.py   (set LOCAL_LLM_BASE_URL / LOCAL_LLM_MODEL if needed)
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


def text_of(result) -> str:
    return "".join(c.text for c in result.content if getattr(c, "text", None))


async def main() -> int:
    env = {k: v for k, v in os.environ.items() if k.startswith("LOCAL_LLM_")}
    server = StdioServerParameters(command="uv", args=["run", "--project", str(ROOT), "local-llm-mcp"], env=env)
    with tempfile.TemporaryDirectory() as d:
        memo = Path(d) / "memo.txt"
        memo.write_text("Meeting memo: the team trip is on November 14, to Hakone. Budget is 30,000 yen per person.\n")
        out = Path(d) / "answer.txt"
        async with Client(server) as client:
            t = text_of(await client.call_tool("local_llm", {"prompt": "Where is the trip? Answer with one word.", "files": [str(memo)]}))
            print("answer:", t)
            assert "Hakone" in t, t
            t = text_of(await client.call_tool("local_llm", {"prompt": "What is the budget?", "files": [str(memo)], "output_path": str(out)}))
            print("notice:", t)
            assert t.startswith("Wrote ") and "30,000" not in t, t
            print("file:", out.read_text().strip())
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
