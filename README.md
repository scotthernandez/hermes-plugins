# Hermes Plugins

Custom plugins for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## Plugins

- **`tool-clean-guard/`** — `pre_tool_call` / `post_tool_call` hygiene: strips leaked LLM XML/ChatML delimiters (`</content>`, `</parameter>`, `</tool_call>`, `<|im_end|>`) and SSH/LXC MOTD banners, validates multi-format syntax (Python AST, JSON, YAML, systemd units, `.env`) before `write_file`, and logs events to `$HERMES_HOME/logs/tool_clean_guard_events.jsonl` (`/tool-clean-guard` command).
- **`triage-router/`** — `llm_request` middleware that classifies reasoning effort (`none`/`low`/`medium`/`high`) using a fast OpenAI-compatible classifier endpoint when `reasoning_effort` is unset or `auto` (`/triage` command).

### Configuration (`triage-router`)

- `TRIAGE_ROUTER_ENDPOINT`: OpenAI-compatible `/v1/chat/completions` URL (default: `http://localhost:1237/v1/chat/completions`)
- `TRIAGE_ROUTER_MODEL`: Optional model name to include in the classifier request payload

## Installation

Copy or symlink a plugin directory into `~/.hermes/plugins/` (or a profile's `plugins/` directory):

```bash
git clone https://github.com/scotthernandez/hermes-plugins.git
cp -r hermes-plugins/tool-clean-guard ~/.hermes/plugins/
cp -r hermes-plugins/triage-router ~/.hermes/plugins/
```
