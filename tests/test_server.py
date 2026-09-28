"""Tests that run without a real LLM: the OpenAI-compatible API is replaced with httpx.MockTransport."""

import json

import httpx
import pytest
from mcp import Client

from local_llm_mcp import server

MEMO = "Meeting memo: the team trip is on November 14, to Hakone. Budget is 30,000 yen per person.\n"


@pytest.fixture
def llm(monkeypatch):
    """Fake /v1/models and /v1/chat/completions. Records the request bodies."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake-model"}]})
        body = json.loads(request.content)
        calls.append(body)
        user = body["messages"][-1]["content"]
        answer = "Hakone" if "Hakone" in user else "unknown"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": answer, "reasoning": "long thoughts"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        })

    real = httpx.AsyncClient
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(server, "_model_cache", None)
    return calls


def text_of(result) -> str:
    return "".join(c.text for c in result.content if getattr(c, "text", None))


async def call(args: dict) -> str:
    async with Client(server.mcp) as client:
        return text_of(await client.call_tool("local_llm", args))


async def test_lists_one_tool():
    async with Client(server.mcp) as client:
        assert [t.name for t in (await client.list_tools()).tools] == ["local_llm"]


async def test_file_goes_to_local_llm_and_only_answer_comes_back(llm, tmp_path):
    memo = tmp_path / "memo.txt"
    memo.write_text(MEMO, encoding="utf-8")
    t = await call({"prompt": "Where is the trip? One word.", "files": [str(memo)]})
    assert t.startswith("Hakone")
    assert "long thoughts" not in t  # thinking text is dropped
    assert "30,000" not in t  # the file itself is not echoed back
    (body,) = llm
    assert body["model"] == "fake-model"  # picked from /v1/models
    assert MEMO in body["messages"][-1]["content"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "none"


async def test_thinking_leaves_reasoning_effort_to_the_server(llm):
    await call({"prompt": "Where is the trip?", "thinking": True})
    (body,) = llm
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in body


async def test_output_path_keeps_answer_out_of_the_result(llm, tmp_path):
    memo = tmp_path / "memo.txt"
    memo.write_text(MEMO, encoding="utf-8")
    out = tmp_path / "out" / "answer.txt"
    t = await call({"prompt": "Where is the trip?", "files": [str(memo)], "output_path": str(out)})
    assert t.startswith("Wrote ") and "Hakone" not in t
    assert out.read_text(encoding="utf-8") == "Hakone"


@pytest.mark.parametrize("case", ["hidden dir", "dotenv", "outside roots", "too large", "overwrite", "hidden output", "missing"])
async def test_refused_before_calling_the_llm(llm, tmp_path, case):
    big = tmp_path / "big.txt"
    big.write_text("a" * (server.MAX_INPUT_CHARS * 4 + 1), encoding="utf-8")
    existing = tmp_path / "existing.txt"
    existing.write_text("keep me", encoding="utf-8")
    args = {
        "hidden dir": {"prompt": "x", "files": ["~/.ssh/id_ed25519"]},
        "dotenv": {"prompt": "x", "files": [str(tmp_path / ".env")]},
        "outside roots": {"prompt": "x", "files": ["/etc/hosts"]},
        "too large": {"prompt": "x", "files": [str(big)]},
        "overwrite": {"prompt": "x", "output_path": str(existing)},
        "hidden output": {"prompt": "x", "output_path": "~/.zshrc.local"},
        "missing": {"prompt": "x", "files": [str(tmp_path / "nope.txt")]},
    }[case]
    t = await call(args)
    assert t.startswith("Refused:"), t
    assert llm == []
    assert existing.read_text(encoding="utf-8") == "keep me"


async def test_symlink_to_hidden_file_is_refused(llm, tmp_path):
    hidden = tmp_path / ".secret"
    hidden.write_text("token", encoding="utf-8")
    link = tmp_path / "innocent.txt"
    link.symlink_to(hidden)
    t = await call({"prompt": "x", "files": [str(link)]})
    assert t.startswith("Refused:") and llm == []
