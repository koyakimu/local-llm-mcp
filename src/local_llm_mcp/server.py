"""MCP server (stdio) that hands work to a local OpenAI-compatible LLM.

File contents are read by this server and sent only to the local LLM, never to the MCP client's model.
The client gets back the local model's answer, or with `output_path` only a notice that the answer was written.
"""

import os
import sys
from pathlib import Path

import httpx
from mcp.server import MCPServer

BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
MODEL = os.environ.get("LOCAL_LLM_MODEL", "")  # empty: use the first model listed by /v1/models
API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "")
MAX_INPUT_CHARS = int(os.environ.get("LOCAL_LLM_MAX_INPUT_CHARS", "60000"))
# The first call may include model loading and a long prefill
TIMEOUT = float(os.environ.get("LOCAL_LLM_TIMEOUT", "600"))

# Where files may be read from and written to. Paths through hidden names (~/.ssh, .env, ~/.config, ...) are refused,
# so that instructions planted in a document cannot make the tool read keys or settings (the answer would carry them
# back to the client) or overwrite files such as ~/.zshrc.
# tempfile.gettempdir() is not used: MCP clients may not pass TMPDIR, and it then falls back to the working directory.
# On macOS the per-user temp dir (/var/folders/...) is asked from the OS for the same reason.
def _darwin_user_temp() -> str:
    if sys.platform != "darwin":
        return ""
    try:
        # _CS_DARWIN_USER_TEMP_DIR from <unistd.h>; Python's os.confstr_names does not list it
        return os.confstr(65537) or ""
    except (ValueError, OSError):
        return ""


_DEFAULT_ROOTS = ":".join(["~", "/tmp", os.environ.get("TMPDIR", ""), _darwin_user_temp()])
ALLOWED_ROOTS = [
    Path(r).expanduser().resolve()
    for r in os.environ.get("LOCAL_LLM_ALLOWED_ROOTS", _DEFAULT_ROOTS).split(":")
    if r
]

mcp = MCPServer("local-llm")
_model_cache: str | None = None


def check_path(raw: str) -> Path:
    """Resolve symlinks, then require the path to be under an allowed root and free of hidden components."""
    p = Path(raw).expanduser().resolve()
    for root in ALLOWED_ROOTS:
        if p.is_relative_to(root):
            if any(part.startswith(".") for part in p.relative_to(root).parts):
                raise ValueError(f"{raw}: hidden files and directories are not allowed")
            return p
    raise ValueError(f"{raw}: outside the allowed directories ({', '.join(map(str, ALLOWED_ROOTS))})")


def read_files(files: list[str], budget: int) -> str:
    paths = [check_path(f) for f in files]
    for p in paths:
        if not p.is_file():
            raise ValueError(f"{p}: not a regular file")
    # Reject by size before reading. UTF-8 uses 1-4 bytes per character, so more than 4x the budget in bytes
    # is certainly over the budget in characters.
    total = sum(p.stat().st_size for p in paths)
    if total > budget * 4:
        raise ValueError(f"files are too large: {total} bytes. Split them into smaller batches")
    parts = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        parts.append(f'<file path="{p}">\n{text}\n</file>')
    return "\n\n".join(parts)


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


async def _model(client: httpx.AsyncClient) -> str:
    global _model_cache
    if MODEL:
        return MODEL
    if _model_cache is None:
        r = await client.get(f"{BASE_URL}/models", headers=_headers())
        r.raise_for_status()
        _model_cache = r.json()["data"][0]["id"]
    return _model_cache


@mcp.tool()
async def local_llm(
    prompt: str,
    files: list[str] | None = None,
    output_path: str | None = None,
    system: str | None = None,
    max_tokens: int = 4096,
    thinking: bool = False,
) -> str:
    """Run a task on a local LLM on this machine. Nothing is sent to the cloud.

    Use it for (1) private data that must not leave the machine: pass file paths in `files` instead of
    reading them yourself; this server reads them and only the local model's answer comes back to you;
    and (2) bulk, mechanical text work (summarizing, classifying, translating, reformatting) where a smaller
    model is good enough. Do not use it for tasks that need strong reasoning or careful code changes.

    Files must be under the home directory or /tmp and must not be hidden (no path component starting with ".").

    Args:
        prompt: The instruction for the local model. Write it self-contained; the model sees nothing else.
        files: Text files to include after the prompt (UTF-8). Total input is capped (~60k characters).
        output_path: If set, the answer is written to this new file and only a short notice is returned,
            so even the answer stays out of the conversation. Existing files are never overwritten.
        system: Optional system prompt.
        max_tokens: Maximum tokens to generate.
        thinking: Enable the model's thinking mode, if it has one (slower, sometimes more accurate).
    """
    content = prompt
    out = None
    try:
        if output_path:
            out = check_path(output_path)
            if out.exists():
                raise ValueError(f"{out}: already exists. Choose a new path; this tool never overwrites files")
        if files:
            content += "\n\n" + read_files(files, MAX_INPUT_CHARS - len(prompt))
    except (ValueError, OSError) as e:
        return f"Refused: {e}"
    if len(content) > MAX_INPUT_CHARS:
        return (
            f"Refused: input is too large: {len(content)} characters (limit {MAX_INPUT_CHARS}). "
            "Split the files into smaller batches and call this tool once per batch."
        )

    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": content})
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            model = await _model(client)
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                # Understood by llama.cpp, vLLM and mlx-lm; ignored by servers without a thinking template
                "chat_template_kwargs": {"enable_thinking": thinking},
            }
            r = await client.post(f"{BASE_URL}/chat/completions", json=body, headers=_headers())
        except httpx.HTTPError as e:
            return f"Could not reach the local LLM at {BASE_URL}: {e}"
    if r.status_code != 200:
        return f"Local LLM returned HTTP {r.status_code}: {r.text[:500]}"
    data = r.json()
    choice = data["choices"][0]
    # Thinking text comes back separately (reasoning / reasoning_content); return only the answer
    answer = choice["message"].get("content") or ""
    usage = data.get("usage", {})
    note = f"[local-llm {model}: prompt {usage.get('prompt_tokens')} tokens, output {usage.get('completion_tokens')} tokens"
    if choice.get("finish_reason") == "length":
        note += ", truncated at max_tokens"
    note += "]"

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            # "x" fails if the file appeared since the check above
            with out.open("x", encoding="utf-8") as fh:
                fh.write(answer)
        except FileExistsError:
            return f"Refused: {out}: already exists. Choose a new path; this tool never overwrites files"
        return f"Wrote {len(answer)} characters to {out}. {note}"
    return f"{answer}\n\n{note}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
