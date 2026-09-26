# local-llm-mcp

An MCP server that lets Claude Code (or any MCP client) hand work to a **local** LLM behind an
OpenAI-compatible API (llama.cpp, llama-swap, mlx-lm, vLLM, LM Studio, Ollama, ...).

Keep using your frontier model for thinking, and send the local model the things it is good for:

- **Private data that must not leave the machine.** Pass file *paths*; this server reads the files and sends them
  only to the local LLM. The client model sees the local model's answer, or with `output_path` only a notice that
  the answer was written to a file.
- **Bulk, mechanical text work** such as summarizing, classifying, translating, or reformatting many files.

## Install

Requires [uv](https://docs.astral.sh/uv/) and a local LLM server with an OpenAI-compatible API.

### Claude Code (plugin)

```sh
claude plugin marketplace add koyakimu/local-llm-mcp
claude plugin install local-llm@koyakimu
```

When the plugin is enabled, Claude Code asks for the base URL (default `http://127.0.0.1:8080/v1`), the model name
(empty = first model from `/v1/models`) and an optional API key, which is kept in secure storage.
Change them later in `/config`.

### Codex (plugin)

```sh
codex plugin marketplace add koyakimu/local-llm-mcp
```

Then install `local-llm` from `/plugins`. The plugin's server uses `http://127.0.0.1:8080/v1` and the first model
from `/v1/models`. Codex stops a tool call after 60 seconds by default, and the first call to a local model can take
longer (model loading, long prompts), so raise the timeout in `~/.codex/config.toml`:

```toml
[mcp_servers.local-llm]
tool_timeout_sec = 600
```

### Any MCP client (manual)

```sh
# Claude Code
claude mcp add --scope user local-llm -e LOCAL_LLM_MODEL=my-model \
  -- uvx --from git+https://github.com/koyakimu/local-llm-mcp local-llm-mcp
# Codex
codex mcp add local-llm --env LOCAL_LLM_MODEL=my-model \
  -- uvx --from git+https://github.com/koyakimu/local-llm-mcp local-llm-mcp
```

## Use

Ask things like:

- "Summarize each file in ~/Documents/minutes/ in three lines with local_llm and write the summaries to ~/Documents/summaries/"
- "Use local_llm to group the errors in ~/logs/app.log by type"

## Tool

`local_llm(prompt, files=None, output_path=None, system=None, max_tokens=4096, thinking=False)`

| Argument | Meaning |
|---|---|
| `prompt` | Instruction for the local model. It sees nothing else, so make it self-contained |
| `files` | UTF-8 text files appended after the prompt |
| `output_path` | Write the answer to this new file and return only a notice. Existing files are never overwritten |
| `system` | Optional system prompt |
| `max_tokens` | Maximum tokens to generate |
| `thinking` | Sends `chat_template_kwargs: {"enable_thinking": ...}` (llama.cpp, vLLM, mlx-lm). Off by default because thinking makes every call much slower |

## Configuration

| Variable | Default | |
|---|---|---|
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible base URL |
| `LOCAL_LLM_MODEL` | first model from `/v1/models` | Model name to request |
| `LOCAL_LLM_API_KEY` | none | Sent as `Authorization: Bearer ...` if set |
| `LOCAL_LLM_MAX_INPUT_CHARS` | `60000` | Limit for prompt + files |
| `LOCAL_LLM_TIMEOUT` | `600` | Seconds; the first call may include model loading |
| `LOCAL_LLM_ALLOWED_ROOTS` | `~`, `/tmp`, `$TMPDIR`, and the macOS per-user temp dir | Colon-separated directories that files may be read from and written to |

## Safety

The paths come from a model that may be reading untrusted documents, so the server limits what it touches:

- Files are read and written only under `LOCAL_LLM_ALLOWED_ROOTS`, after resolving symlinks.
- Any path with a hidden component (`~/.ssh`, `.env`, `~/.config`, ...) is refused, so keys and settings cannot be
  pulled into an answer and dotfiles cannot be overwritten.
- File sizes are checked before reading; `output_path` never overwrites an existing file.

The answer itself can still contain parts of the input. Use `output_path` when even the answer should stay local.

## Development

```sh
uv run --group dev pytest               # no LLM needed; the API is mocked
uv run scripts/check_live.py            # against a running local LLM
```

## License

MIT
