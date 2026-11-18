"""Dynamic Request Modifier Plugin — Dynamically modifies outbound LLM request parameters via an auxiliary classifier."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger("hermes.plugin.dynamic_request_modifier")

MODIFIER_ENDPOINT = os.environ.get("DYNAMIC_REQUEST_MODIFIER_ENDPOINT", "").strip()
MODIFIER_MODEL = os.environ.get("DYNAMIC_REQUEST_MODIFIER_MODEL", "").strip()
AUX_TASK = "dynamic_request_modifier"
PLUGIN_DIR = Path(__file__).parent
FEW_SHOT_FILE = PLUGIN_DIR / "triage_few_shot.json"
AUDIT_LOG_FILE = Path(get_hermes_home()) / "logs" / "dynamic_request_modifier_audit.jsonl"

_PLUGIN_CTX: Any = None
_TURN_EFFORT_CACHE: Dict[tuple[str, str], str] = {}
_MAX_TURN_CACHE = 256

FAST_NONE_PROMPTS = {
    "hi", "hello", "hey", "yes", "no", "ok", "okay", "thanks", "thank you",
    "status", "ping", "whoami", "clear", "exit", "quit", "help"
}


def _is_default_profile() -> bool:
    home = str(get_hermes_home())
    return "/profiles/" not in home


def _load_few_shot_examples() -> str:
    if not FEW_SHOT_FILE.exists():
        return ""
    try:
        with open(FEW_SHOT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            lines = [f"Input: {item['prompt']}\nClassification: {item['effort']}" for item in data[:8]]
            return "\n\n".join(lines)
    except Exception as exc:
        logger.debug("Failed to load few-shot examples: %s", exc)
        return ""


def _parse_effort(raw_text: str) -> str:
    word = (raw_text or "").strip().upper()
    for candidate in ("NONE", "LOW", "MEDIUM", "HIGH"):
        if candidate in word:
            return candidate.lower()
    return "medium"


def _classify_effort(prompt: str) -> tuple[str, float]:
    few_shot = _load_few_shot_examples()
    system_msg = (
        "You are a strict, ultra-fast intent classifier. Classify the user prompt's required reasoning effort into exactly ONE word: "
        "NONE (greetings, simple yes/no, trivial facts), "
        "LOW (quick commands, git status, file cat, single tool check), "
        "MEDIUM (standard coding, writing functions, general tasks), or "
        "HIGH (multi-step planning, systems architecture, debugging traces, deep research). "
        "Reply with ONLY that single uppercase word."
    )
    user_content = f"{few_shot}\n\nInput: {prompt}\nClassification:" if few_shot else f"Input: {prompt}\nClassification:"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    if not MODIFIER_ENDPOINT:
        if _PLUGIN_CTX is None or not hasattr(_PLUGIN_CTX, "llm"):
            raise RuntimeError("PluginContext.llm unavailable and DYNAMIC_REQUEST_MODIFIER_ENDPOINT is unset")
        t0 = time.time()
        res = _PLUGIN_CTX.llm.complete(
            messages,
            temperature=0.0,
            max_tokens=5,
            timeout=1.8,
            purpose="request_effort_classification",
            task=AUX_TASK,
        )
        dt = (time.time() - t0) * 1000.0
        return _parse_effort(res.text), dt

    payload: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": 5,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
    }
    if MODIFIER_MODEL:
        payload["model"] = MODIFIER_MODEL

    req = urllib.request.Request(
        MODIFIER_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1.8) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        dt = (time.time() - t0) * 1000.0
        return _parse_effort(res["choices"][0]["message"]["content"]), dt


def _log_triage(prompt: str, effort: str, dt_ms: float, source: str, session_id: str = "") -> None:
    try:
        AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "prompt": prompt[:200],
            "effort": effort,
            "latency_ms": round(dt_ms, 2),
            "source": source
        }
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.debug("Failed to write triage audit log: %s", exc)


def _on_llm_request(request: Dict[str, Any], **context: Any) -> Optional[Dict[str, Any]]:
    # 1. Profile Isolation: Run ONLY for default profile
    if not _is_default_profile():
        return None

    # 2. Honor explicit reasoning if set by user or agent config
    if request.get("reasoning_effort") not in (None, "", "auto"):
        return None
    if (request.get("extra_body") or {}).get("reasoning_effort"):
        return None

    session_id = str(context.get("session_id") or "")
    turn_id = str(context.get("turn_id") or "")
    cache_key = (session_id, turn_id) if turn_id else None
    if cache_key and cache_key in _TURN_EFFORT_CACHE:
        request["reasoning_effort"] = _TURN_EFFORT_CACHE[cache_key]
        return {"request": request}

    # 3. Extract the last user message
    user_prompt = ""
    messages = request.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_prompt = content.strip()
            elif isinstance(content, list):
                parts = [
                    part.get("text", "").strip()
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                user_prompt = " ".join(p for p in parts if p)
            break

    if not user_prompt:
        return None

    clean_lower = user_prompt.lower()
    # 4. Fast path for trivial greetings and commands (0ms)
    if clean_lower in FAST_NONE_PROMPTS or len(clean_lower) < 8:
        effort = "none"
        _log_triage(user_prompt, effort, 0.0, "fast_path", session_id)
    else:
        # 5. Fast classification via configured auxiliary/classifier model
        try:
            effort, dt = _classify_effort(user_prompt)
            _log_triage(user_prompt, effort, dt, "classifier_endpoint", session_id)
        except Exception as exc:
            logger.warning("Dynamic request classification failed (%s), falling back to medium", exc)
            effort = "medium"

    if cache_key:
        if len(_TURN_EFFORT_CACHE) >= _MAX_TURN_CACHE:
            _TURN_EFFORT_CACHE.clear()
        _TURN_EFFORT_CACHE[cache_key] = effort

    request["reasoning_effort"] = effort
    return {"request": request}


def _handle_triage_command(raw_args: str) -> str:
    argv = raw_args.strip().split()
    cmd = argv[0] if argv else "status"

    if cmd == "test" and len(argv) > 1:
        test_prompt = " ".join(argv[1:])
        effort, dt = _classify_effort(test_prompt)
        return f"Triage test result for: '{test_prompt}'\nEffort: **{effort.upper()}** (in {dt:.1f} ms)"

    if not AUDIT_LOG_FILE.exists():
        return "Dynamic Request Modifier is active. No audit entries recorded yet."

    counts: Dict[str, int] = {"none": 0, "low": 0, "medium": 0, "high": 0}
    total = 0
    recent = []
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                eff = entry.get("effort", "medium")
                counts[eff] = counts.get(eff, 0) + 1
                total += 1
                recent.append(entry)
            except Exception:
                continue

    out = [f"**Dynamic Request Modifier Summary** (Total Classified: {total})"]
    for k, v in counts.items():
        pct = (v / total * 100) if total else 0
        out.append(f"- {k.upper():6s}: {v} ({pct:.1f}%)")

    out.append("\n**Recent Classifications:**")
    for r in recent[-5:]:
        out.append(f"- [{r.get('effort', '').upper()}] ({r.get('latency_ms', 0)}ms): {r.get('prompt', '')[:60]}...")

    return "\n".join(out)


def register(ctx) -> None:
    global _PLUGIN_CTX
    _PLUGIN_CTX = ctx
    if hasattr(ctx, "register_auxiliary_task"):
        ctx.register_auxiliary_task(
            AUX_TASK,
            display_name="Dynamic Request Modifier",
            description="Fast classifier for dynamic outbound LLM request parameter tuning.",
            inherit_from="triage_specifier",
            defaults={"timeout": 2},
        )
    ctx.register_middleware("llm_request", _on_llm_request)
    ctx.register_command("triage", handler=_handle_triage_command, description="Inspect dynamic request modifier statistics or test prompt classification.")
