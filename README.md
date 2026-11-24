# Hermes Plugins

Custom plugins for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

## Plugins

- **`tool-clean-guard/`** — `pre_tool_call` / `post_tool_call` hygiene: strips leaked LLM XML/ChatML delimiters (`</content>`, `</parameter>`, `</tool_call>`, `<|im_end|>`) and SSH/LXC MOTD banners, validates multi-format syntax (Python AST, JSON, YAML, systemd units, `.env`) before `write_file`, and logs events to `$HERMES_HOME/logs/tool_clean_guard_events.jsonl` (`/tool-clean-guard` command).
- **`dynamic-request-modifier/`** — `llm_request` middleware that classifies reasoning effort (`none`/`low`/`medium`/`high`) via Hermes's auxiliary model (`auxiliary.dynamic_request_modifier`) when `reasoning_effort` is unset or `auto`, injects per-request `reasoning_budget` + `chat_template_kwargs.enable_thinking` for local/llama-server providers, and steps down effort on clean tool-loop follow-ups while keeping/escalating effort after tool errors (`/triage` command).

### Configuration (`dynamic-request-modifier`)

By default, `dynamic-request-modifier` registers an `auxiliary.dynamic_request_modifier` task (inheriting from `triage_specifier` when `plugins.entries.dynamic-request-modifier.llm.allow_task_override: true`, or configurable directly under `auxiliary.dynamic_request_modifier` in `config.yaml`) and routes through Hermes's configured auxiliary/active model via `ctx.llm`.

Optional direct-endpoint override:
- `DYNAMIC_REQUEST_MODIFIER_ENDPOINT`: OpenAI-compatible `/v1/chat/completions` URL (e.g. `http://localhost:1237/v1/chat/completions`) to bypass `ctx.llm`
- `DYNAMIC_REQUEST_MODIFIER_MODEL`: Optional model name to include when using `DYNAMIC_REQUEST_MODIFIER_ENDPOINT`

## Installation

Copy or symlink a plugin directory into `~/.hermes/plugins/` (or a profile's `plugins/` directory):

```bash
git clone https://github.com/scotthernandez/hermes-plugins.git
ln -sfn /path/to/hermes-plugins/tool-clean-guard ~/.hermes/plugins/tool-clean-guard
ln -sfn /path/to/hermes-plugins/dynamic-request-modifier ~/.hermes/plugins/dynamic-request-modifier
```
