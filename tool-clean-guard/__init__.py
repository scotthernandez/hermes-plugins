"""Tool Clean Guard Plugin — Pre-write/pre-exec XML/ChatML delimiter stripping, MOTD stripping, multi-format validation, and audit logging."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("hermes.plugin.tool_clean_guard")

_LEAKED_TRAILING_DELIMITERS_RE = re.compile(
    r"(?:\r?\n)*\s*(?:</content>|</parameter>|</tool_call>|<\|im_end\|>)\s*$"
)
_LEAKED_PATH_TAG_RE = re.compile(
    r"</?(?:parameter|function|tool_call|content)\b|<\|im_end\|>",
    re.IGNORECASE,
)
_MOTD_CONTAINER_RE = re.compile(r"(?:Ubuntu|Debian|Alpine|Proxmox).*LXC Container")
_MOTD_PROVIDER_RE = re.compile(r"Provided by:.*community-scripts ORG")
_MOTD_FIELD_RE = re.compile(r"^(?:\s*(?:OS|Hostname|IP Address):|\s*$)")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ENV_LINE_RE = re.compile(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*=.*$")
_SYSTEMD_SECTION_RE = re.compile(r"^\[[A-Za-z0-9_.-]+\]$")
_SYSTEMD_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_MARKUP_EXTENSIONS = (
    ".html", ".htm", ".xml", ".svg", ".md", ".mdx", ".jinja", ".jinja2", ".j2"
)
_SYSTEMD_EXTENSIONS = (".service", ".container", ".timer")


def _events_file() -> Path:
    return Path(get_hermes_home()) / "logs" / "tool_clean_guard_events.jsonl"


def _record_event(event: str, tool_name: str, *, path: str = "", session_id: str = "", detail: str = "") -> None:
    logger.warning("tool-clean-guard event=%s tool=%s path=%s session=%s detail=%s",
                   event, tool_name, path or "-", session_id or "-", detail[:160])
    try:
        log_path = _events_file()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "tool": tool_name,
            "path": path,
            "session_id": session_id,
            "detail": detail[:300],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.debug("tool-clean-guard failed to write audit log: %s", exc)


def _strip_leaked_delimiters(text: str) -> str:
    while True:
        cleaned = _LEAKED_TRAILING_DELIMITERS_RE.sub("", text)
        if cleaned == text:
            break
        text = cleaned + ("\n" if text.endswith("\n") and cleaned else "")
    return text


def _strip_leading_motd_banner(text: str) -> str:
    lines = text.splitlines(keepends=True)
    head = lines[:15]
    if not (any(_MOTD_CONTAINER_RE.search(ln) for ln in head) and any(_MOTD_PROVIDER_RE.search(ln) for ln in head)):
        return text
    last_motd_idx = -1
    for i, line in enumerate(head):
        clean = _ANSI_ESCAPE_RE.sub("", line).strip()
        if (
            not clean
            or _MOTD_CONTAINER_RE.search(clean)
            or _MOTD_PROVIDER_RE.search(clean)
            or _MOTD_FIELD_RE.match(clean)
            or "IP Address:" in clean
            or "Hostname:" in clean
            or "OS:" in clean
        ):
            last_motd_idx = i
        else:
            break
    if last_motd_idx < 0:
        return text
    return "".join(lines[last_motd_idx + 1:])


def _is_env_path(path: str) -> bool:
    base = os.path.basename(path).lower()
    return base == ".env" or base.startswith(".env.") or base.endswith(".env")


def _validate_env_content(path: str, content: str) -> Optional[str]:
    in_quote: Optional[str] = None
    for lineno, raw_line in enumerate(content.splitlines(), 1):
        if "\x1b" in raw_line or _ANSI_ESCAPE_RE.search(raw_line):
            return (
                f"Refusing to write ANSI escape sequences into {path} (line {lineno}). "
                "Strip terminal escape codes before writing .env files."
            )
        stripped = raw_line.strip()
        if in_quote is not None:
            if stripped.endswith(in_quote) and not stripped.endswith("\\" + in_quote):
                in_quote = None
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if not _ENV_LINE_RE.match(stripped):
            return (
                f"Refusing to write invalid .env syntax to {path} "
                f"(line {lineno}: {stripped[:60]!r}). Every non-comment line must be KEY=VALUE."
            )
        val = stripped.split("=", 1)[1].lstrip()
        if val.startswith(('"', "'")) and re.search(r"(?<!\\)" + re.escape(val[0]), val[1:]) is None:
            in_quote = val[0]
    return None


def _validate_systemd_content(path: str, content: str) -> Optional[str]:
    seen_section = False
    in_continuation = False
    for lineno, raw_line in enumerate(content.splitlines(), 1):
        stripped = raw_line.strip()
        if in_continuation:
            in_continuation = stripped.endswith("\\")
            continue
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if _SYSTEMD_SECTION_RE.match(stripped):
            seen_section = True
            continue
        if not seen_section:
            return (
                f"Refusing to write invalid systemd/Quadlet unit to {path} "
                f"(line {lineno}: expected [Section] header before {stripped[:60]!r})."
            )
        if "=" not in stripped:
            return (
                f"Refusing to write invalid systemd/Quadlet directive to {path} "
                f"(line {lineno}: missing '=' in {stripped[:60]!r})."
            )
        key = stripped.split("=", 1)[0].strip()
        if not _SYSTEMD_KEY_RE.match(key):
            return (
                f"Refusing to write invalid systemd/Quadlet key to {path} "
                f"(line {lineno}: {key!r})."
            )
        in_continuation = stripped.endswith("\\")
    if content.strip() and not seen_section:
        return f"Refusing to write systemd/Quadlet unit to {path}: no [Section] header found."
    return None


def _sanitize_and_validate_write_content(path: str, content: str) -> tuple[str, Optional[str]]:
    """Strip trailing leaked XML tool delimiters / leading MOTD banners and validate structured formats pre-write."""
    if not isinstance(content, str) or not isinstance(path, str) or not path:
        return content, None

    lower_path = path.lower()
    if not lower_path.endswith(_MARKUP_EXTENSIONS):
        content = _strip_leading_motd_banner(content)
        content = _strip_leaked_delimiters(content)

    if lower_path.endswith(".py") and content.strip():
        try:
            ast.parse(content, filename=path)
        except SyntaxError as exc:
            bad_line = (exc.text or "").strip()
            return content, (
                f"Refusing to write invalid Python syntax to {path} "
                f"(line {exc.lineno}: {exc.msg}: {bad_line!r}). "
                "Fix the Python syntax error in your content before calling write_file."
            )
    elif _is_env_path(path):
        err = _validate_env_content(path, content)
        if err:
            return content, err
    elif lower_path.endswith(".json") and content.strip():
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            return content, (
                f"Refusing to write invalid JSON syntax to {path} "
                f"(line {exc.lineno}, col {exc.colno}: {exc.msg})."
            )
    elif lower_path.endswith(_SYSTEMD_EXTENSIONS) and content.strip():
        err = _validate_systemd_content(path, content)
        if err:
            return content, err

    return content, None


def _on_pre_tool_call(tool_name: str, args: Dict[str, Any], **kwargs: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(args, dict):
        return None
    session_id = str(kwargs.get("session_id") or "")

    if tool_name in ("write_file", "patch"):
        path = args.get("path")
        if isinstance(path, str) and _LEAKED_PATH_TAG_RE.search(path):
            msg = f"Refusing {tool_name}: path contains leaked XML/ChatML delimiter ({path!r})."
            _record_event("path_blocked", tool_name, path=path, session_id=session_id, detail=msg)
            return {"action": "block", "message": msg}

    if tool_name == "write_file":
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            return None
        cleaned, err = _sanitize_and_validate_write_content(path, content)
        if err:
            _record_event("syntax_blocked", tool_name, path=path, session_id=session_id, detail=err)
            return {"action": "block", "message": err}
        if cleaned != content:
            _record_event("content_sanitized", tool_name, path=path, session_id=session_id,
                          detail=f"stripped {len(content) - len(cleaned)} chars")
            return {"action": "modify", "args": {"content": cleaned}}
        return None

    if tool_name == "execute_code":
        code = args.get("code")
        if not isinstance(code, str):
            return None
        cleaned = _strip_leaked_delimiters(code)
        if cleaned != code:
            _record_event("delimiter_stripped", tool_name, session_id=session_id,
                          detail=f"stripped {len(code) - len(cleaned)} chars")
            return {"action": "modify", "args": {"code": cleaned}}
        return None

    if tool_name == "patch":
        path = args.get("path")
        if isinstance(path, str) and path.lower().endswith(_MARKUP_EXTENSIONS):
            return None
        updates: Dict[str, str] = {}
        for field in ("new_string", "patch"):
            val = args.get(field)
            if isinstance(val, str):
                cleaned = _strip_leaked_delimiters(val)
                if cleaned != val:
                    updates[field] = cleaned
        if updates:
            _record_event("delimiter_stripped", tool_name, path=str(path or ""), session_id=session_id,
                          detail=f"fields={','.join(sorted(updates))}")
            return {"action": "modify", "args": updates}
        return None

    return None


def _on_post_tool_call(tool_name: str, result: Any = None, **kwargs: Any) -> None:
    if tool_name != "execute_code":
        return
    text = str(result or "")
    if "SyntaxError:" in text:
        session_id = str(kwargs.get("session_id") or "")
        line = next((ln.strip() for ln in text.splitlines() if "SyntaxError:" in ln), "SyntaxError")
        _record_event("execute_code_syntax_error", tool_name, session_id=session_id, detail=line)


def _handle_write_guard_command(raw_args: str = "") -> str:
    log_path = _events_file()
    if not log_path.exists():
        return f"tool-clean-guard is active. No events recorded yet ({log_path})."
    counts: Dict[str, int] = {}
    recent = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                ev = str(entry.get("event") or "unknown")
                counts[ev] = counts.get(ev, 0) + 1
                recent.append(entry)
            except Exception:
                continue
    out = [f"**Tool Clean Guard Summary** (`{log_path}`, total: {sum(counts.values())})"]
    for k, v in sorted(counts.items()):
        out.append(f"- `{k}`: {v}")
    if recent:
        out.append("\n**Recent Events:**")
        for r in recent[-5:]:
            out.append(f"- `{r.get('timestamp', '')}` [{r.get('event')}] `{r.get('tool')}` {r.get('path') or '-'} — {r.get('detail', '')[:80]}")
    return "\n".join(out)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_command(
        "tool-clean-guard",
        handler=_handle_write_guard_command,
        description="Inspect tool-clean-guard delimiter stripping and syntax-block metrics.",
    )
