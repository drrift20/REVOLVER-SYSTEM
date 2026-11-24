"""
Revolver System — Self-Healing, Multi-Trigger, Judge Revolver Engine.

A standalone, fault-tolerant AI orchestration library for concurrent model racing,
adaptive health scoring, circuit breakers, and automated draft repair.

Key Features:
- Zero hardcoded keys: Pass your own API keys for any OpenAI-compatible provider
  (Groq, Gemini, OpenRouter, OpenAI, Mistral, Cerebras, or custom endpoints).
- Dynamic Chamber Generation: Auto-discovers models directly from provider endpoints
  or accepts custom model lists to build chambers on the fly.
- Dual-Trigger Concurrent Race: Races top-2 healthy chambers concurrently on a pooled
  thread executor. First valid response wins; slower loser is cancelled.
- Adaptive Health & Cooldowns: Per-chamber health tracking (0-100), fast-response boosts,
  and granular penalties on 429, 5xx, timeouts, or model errors.
- The Judge Revolver: Conditional self-repair engine that detects malformed drafts
  (markdown tables, prompt leakage, cutoff text) and prompts a healthy chamber to fix it.
- Default English Judge + fully customizable judge instructions and system prompts.
"""

import concurrent.futures
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger("revolver")

# ---------------------------------------------------------------------------
# Default Tuning Constants
# ---------------------------------------------------------------------------
DEFAULT_FAST_THRESHOLD = 4.0             # seconds: responses under this gain fast boost
DEFAULT_FAST_BOOST = 12.0                # health points added on fast success
DEFAULT_SLOW_BOOST = 6.0                 # health points added on slower success
DEFAULT_RECOVERY_PER_SEC = 100.0 / 240.0 # passive recovery: full 0->100 in ~4 minutes
DEFAULT_TIMEOUT_SECONDS = 15

# Outcome penalties: outcome_name -> (health_penalty, cooldown_seconds)
DEFAULT_OUTCOME_PENALTY = {
    "429": (45.0, 30.0),
    "5xx": (22.0, 15.0),
    "timeout": (18.0, 12.0),
    "network": (12.0, 10.0),
    "parse": (12.0, 10.0),
    "empty": (15.0, 10.0),
    "httperr": (25.0, 30.0),
    "model_error": (60.0, 3600.0),  # 400/404 bad/dead model -> heavy penalty + 1h cooldown
    "exception": (10.0, 8.0),
}

# Generic corruption markers for draft validation
DEFAULT_CORRUPTION_MARKERS = (
    "as an ai language model",
    "i'm sorry, but i cannot",
    "i cannot fulfill",
    "system prompt",
    "[truncated]",
    "response truncated",
    "cut off",
    ">>",
    "<<",
)

# Standard English Default Judge Instruction (General-purpose, zero bot-specific jargon)
DEFAULT_JUDGE_INSTRUCTION = (
    "A draft reply violated an output rule ({reason}). "
    "Rewrite it into a clean, valid reply. Stay concise, preserve factual accuracy, "
    "and never use markdown tables or pipe characters. "
    "Output ONLY the corrected message text, with no preamble or explanation."
)

# Provider Endpoint Catalog
KNOWN_PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "models_url": "https://api.groq.com/openai/v1/models",
        "default_models": [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "models_url": "https://generativelanguage.googleapis.com/v1beta/openai/models",
        "default_models": [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ],
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "models_url": "https://openrouter.ai/api/v1/models",
        "default_models": [
            "openrouter/free",
            "google/gemma-2-9b-it:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ],
        "extra_headers": {
            "HTTP-Referer": "https://github.com/revolver-system",
            "X-Title": "Revolver System",
        },
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "models_url": "https://api.openai.com/v1/models",
        "default_models": [
            "gpt-4o-mini",
            "gpt-4o",
        ],
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "models_url": "https://api.mistral.ai/v1/models",
        "default_models": [
            "mistral-small-latest",
        ],
    },
    "cerebras": {
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "models_url": "https://api.cerebras.ai/v1/models",
        "default_models": [
            "llama3.3-70b",
            "llama3.1-8b",
        ],
    },
}


class Revolver:
    """
    Self-Healing, Multi-Trigger, Judge Revolver Engine.

    Manages a dynamic pool of AI chambers across multiple providers/models,
    races them concurrently, tracks per-chamber health with auto-cooldowns,
    and repairs corrupted outputs with the Judge Revolver.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful, concise, and intelligent AI assistant.",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_race_workers: int = 4,
        judge_max_attempts: int = 2,
        judge_instruction: Optional[str] = None,
        corruption_markers: Optional[Tuple[str, ...]] = None,
        session_pool_size: int = 8,
    ):
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.judge_max_attempts = judge_max_attempts
        self.judge_instruction = judge_instruction or DEFAULT_JUDGE_INSTRUCTION
        self.corruption_markers = corruption_markers or DEFAULT_CORRUPTION_MARKERS

        # Thread safety & state storage
        self._health_lock = threading.Lock()
        self._chamber_state: Dict[str, Dict[str, Any]] = {}
        self._chambers: List[Dict[str, Any]] = []

        # Pooled HTTP session for connection reuse
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=session_pool_size)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Concurrency race executor
        self._race_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_race_workers, thread_name_prefix="revolver-race"
        )

    # ------------------------------------------------------------------
    # Chamber & Provider Registration
    # ------------------------------------------------------------------

    def add_chamber(
        self,
        name: str,
        url: str,
        api_key: str,
        model: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Register a single AI chamber. Automatically initializes health state.
        """
        headers = {"Authorization": f"Bearer {api_key}"}
        if extra_headers:
            headers.update(extra_headers)

        chamber = {
            "name": name,
            "url": url,
            "key": api_key,
            "model": model,
            "headers": headers,
        }

        with self._health_lock:
            # Check for existing chamber with same name to avoid duplicate registration
            existing_idx = next((i for i, c in enumerate(self._chambers) if c["name"] == name), None)
            if existing_idx is not None:
                self._chambers[existing_idx] = chamber
            else:
                self._chambers.append(chamber)
            # Lazily/eagerly initialize health state
            self._get_state_locked(name)

        log.info("Registered chamber: %s (model: %s)", name, model)

    def discover_models(self, models_url: str, api_key: str) -> List[str]:
        """
        Query provider's /models endpoint to discover available models.
        """
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            resp = self._session.get(models_url, headers=headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                raw_models = data.get("data") or data.get("models") or []
                model_ids = []
                for item in raw_models:
                    m_id = item.get("id") or item.get("name") if isinstance(item, dict) else str(item)
                    if m_id:
                        # Normalize format if needed (e.g. 'models/gemini-pro' -> 'gemini-pro')
                        if m_id.startswith("models/"):
                            m_id = m_id.replace("models/", "")
                        model_ids.append(m_id)
                return model_ids
        except Exception as exc:
            log.warning("Model discovery failed on %s: %s", models_url, exc)
        return []

    def add_provider(
        self,
        provider: str,
        api_key: str,
        models: Optional[List[str]] = None,
        url: Optional[str] = None,
        models_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        discover_models: bool = False,
        max_chambers: int = 5,
    ) -> List[str]:
        """
        Add a provider with one or multiple models.
        If `models` is omitted and `discover_models=True`, queries the provider API
        to find available models and creates chambers for them.

        Returns list of chamber names created.
        """
        p_name = provider.lower().strip()
        provider_info = KNOWN_PROVIDERS.get(p_name, {})

        target_url = url or provider_info.get("url")
        if not target_url:
            raise ValueError(f"Unknown provider '{provider}' and no url provided.")

        headers = provider_info.get("extra_headers", {}).copy()
        if extra_headers:
            headers.update(extra_headers)

        target_models: List[str] = []
        if models:
            target_models = list(models)
        elif discover_models:
            m_url = models_url or provider_info.get("models_url")
            if m_url:
                discovered = self.discover_models(m_url, api_key)
                if discovered:
                    # Filter out non-chat / whisper / embedding models if possible
                    filtered = [
                        m for m in discovered
                        if not any(skip in m.lower() for skip in ("embed", "whisper", "tts", "moderation", "dall-e", "audio"))
                    ]
                    target_models = filtered[:max_chambers] if filtered else discovered[:max_chambers]

        # Fallback to known default models if discovery returned nothing
        if not target_models:
            target_models = provider_info.get("default_models", [])

        if not target_models:
            raise ValueError(f"No models specified or discovered for provider '{provider}'.")

        created_chambers = []
        for model in target_models:
            chamber_name = f"{p_name.capitalize()} {model}"
            self.add_chamber(
                name=chamber_name,
                url=target_url,
                api_key=api_key,
                model=model,
                extra_headers=headers,
            )
            created_chambers.append(chamber_name)

        return created_chambers

    # ------------------------------------------------------------------
    # Health Scoring & Chamber Ranking
    # ------------------------------------------------------------------

    def _get_state_locked(self, name: str) -> Dict[str, Any]:
        """Fetch or initialize a chamber's mutable health record. Caller holds lock."""
        state = self._chamber_state.get(name)
        if state is None:
            state = {
                "health": 100.0,
                "cooldown_until": 0.0,
                "last_used": 0.0,
                "last_update": time.time(),
                "successes": 0,
                "failures": 0,
            }
            self._chamber_state[name] = state
        return state

    def _decay_locked(self, state: Dict[str, Any], now: float) -> None:
        """Heal a chamber's health back toward 100 based on elapsed time."""
        last = state["last_update"]
        if now > last:
            state["health"] = min(100.0, state["health"] + DEFAULT_RECOVERY_PER_SEC * (now - last))
            state["last_update"] = now

    def _apply_outcome(self, chamber: Dict[str, Any], outcome: Tuple[str, Any]) -> None:
        """Fold request outcome into chamber's health and cooldown."""
        name = chamber["name"]
        kind = outcome[0]
        now = time.time()
        with self._health_lock:
            state = self._get_state_locked(name)
            self._decay_locked(state, now)
            if kind == "ok":
                elapsed = outcome[1] if len(outcome) > 1 and outcome[1] is not None else 0.0
                boost = DEFAULT_FAST_BOOST if elapsed <= DEFAULT_FAST_THRESHOLD else DEFAULT_SLOW_BOOST
                state["health"] = min(100.0, state["health"] + boost)
                state["cooldown_until"] = 0.0
                state["last_used"] = now
                state["successes"] += 1
            else:
                penalty, cooldown = DEFAULT_OUTCOME_PENALTY.get(kind, (10.0, 8.0))
                state["health"] = max(0.0, state["health"] - penalty)
                state["cooldown_until"] = max(state["cooldown_until"], now + cooldown)
                state["failures"] += 1
            state["last_update"] = now

    def rank_chambers(self, now: Optional[float] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (healthy, cooled) lists ranked by health, then least-recently-used."""
        if now is None:
            now = time.time()
        healthy = []
        cooled = []
        with self._health_lock:
            for chamber in self._chambers:
                state = self._get_state_locked(chamber["name"])
                self._decay_locked(state, now)
                entry = (state["health"], state["last_used"], chamber)
                if state["cooldown_until"] > now:
                    cooled.append(entry)
                else:
                    healthy.append(entry)

        # Sort descending by health, then ascending by last_used (least recently used)
        healthy.sort(key=lambda item: (-item[0], item[1]))
        cooled.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in healthy], [item[2] for item in cooled]

    # ------------------------------------------------------------------
    # Network Call & Concurrent Race
    # ------------------------------------------------------------------

    def _call_chamber(
        self,
        chamber: Dict[str, Any],
        messages: List[Dict[str, str]],
        timeout: Optional[int] = None,
    ) -> Tuple[Optional[str], Tuple[str, Any]]:
        """Fire a single chamber request. Returns (content_or_None, outcome_tuple)."""
        name = chamber["name"]
        headers = chamber["headers"].copy()
        headers["Content-Type"] = "application/json"
        payload = {
            "model": chamber["model"],
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.7,
        }
        req_timeout = timeout or self.timeout
        started = time.monotonic()
        try:
            resp = self._session.post(chamber["url"], headers=headers, json=payload, timeout=req_timeout)
        except requests.exceptions.Timeout:
            return None, ("timeout", None)
        except requests.exceptions.RequestException as exc:
            log.warning("Chamber %s network exception: %s", name, exc)
            return None, ("network", None)

        elapsed = time.monotonic() - started

        if resp.status_code == 200:
            try:
                data = resp.json()
                content = (data["choices"][0]["message"].get("content") or "").strip()
            except (KeyError, IndexError, ValueError, AttributeError):
                log.warning("Chamber %s returned unexpected format", name)
                return None, ("parse", None)
            if content:
                return content, ("ok", elapsed)
            return None, ("empty", None)

        if resp.status_code == 429:
            log.warning("Chamber %s hit rate limit (429)", name)
            return None, ("429", None)
        if resp.status_code in (400, 404):
            log.warning("Chamber %s model error (%d): %s", name, resp.status_code, resp.text[:140])
            return None, ("model_error", resp.status_code)
        if resp.status_code >= 500:
            log.warning("Chamber %s failed with HTTP %d", name, resp.status_code)
            return None, ("5xx", None)

        return None, ("httperr", resp.status_code)

    def _race(
        self,
        chambers: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        deadline: float,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Run top chambers concurrently; first valid non-empty answer wins."""
        if not chambers:
            return None, None
        if len(chambers) == 1:
            content, outcome = self._call_chamber(chambers[0], messages)
            self._apply_outcome(chambers[0], outcome)
            return content, (chambers[0]["name"] if content else None)

        future_map = {
            self._race_executor.submit(self._call_chamber, chamber, messages): chamber
            for chamber in chambers
        }
        winner_content = None
        winner_name = None
        try:
            remaining = max(0.1, min(deadline - time.monotonic(), self.timeout + 2.0))
            for future in concurrent.futures.as_completed(future_map, timeout=remaining):
                chamber = future_map[future]
                try:
                    content, outcome = future.result()
                except Exception as exc:
                    log.warning("Racer %s raised: %s", chamber["name"], exc)
                    content, outcome = None, ("exception", None)
                self._apply_outcome(chamber, outcome)
                if content:
                    winner_content = content
                    winner_name = chamber["name"]
                    # Cancel slower competitors immediately
                    for other in future_map:
                        if other is not future:
                            other.cancel()
                    break
        except concurrent.futures.TimeoutError:
            log.warning("Revolver race timed out across %d chambers", len(chambers))
            for future in future_map:
                future.cancel()

        return winner_content, winner_name

    # ------------------------------------------------------------------
    # Sanitization, Normalization & Output Validation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_text(text: str) -> str:
        """Repair encoding glitches (NBSP, non-breaking hyphens, mangled dash ranges)."""
        if not text:
            return ""
        text = text.replace("\u2011", "-")
        text = text.replace("\u202f", " ").replace("\u00a0", " ")
        chars = list(text)
        for i in range(1, len(chars) - 1):
            if chars[i] != "?":
                continue
            prev, nxt = chars[i - 1], chars[i + 1]
            if prev.isdigit() and (nxt.isdigit() or nxt == "-"):
                chars[i] = "-"
            elif nxt.isdigit() and prev == "-":
                chars[i] = ""
        text = "".join(chars)
        while "--" in text:
            text = text.replace("--", "-")
        return text

    @staticmethod
    def has_table(text: str) -> bool:
        """True if text contains markdown table syntax or pipe-delimited rows."""
        for line in text.splitlines():
            if line.count("|") >= 2:
                return True
            if "|" in line and "-" in line:
                residue = line.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
                if residue == "":
                    return True
        return False

    def sanitize_draft(self, text: str) -> str:
        """Normalize encoding, strip markdown table rows/separators, trim blank lines."""
        if not text:
            return ""
        normalized = self.normalize_text(text)
        kept = []
        for line in normalized.splitlines():
            if line.count("|") >= 2:
                continue
            if "|" in line and "-" in line:
                residue = line.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
                if residue == "":
                    continue
            kept.append(line)
        cleaned = "\n".join(kept).strip()
        while "\n\n\n" in cleaned:
            cleaned = cleaned.replace("\n\n\n", "\n\n")
        return cleaned

    def validate_draft(self, text: str) -> Tuple[bool, str]:
        """Return (is_valid, reason)."""
        stripped = self.normalize_text(text).strip() if text else ""
        if not stripped:
            return False, "empty response"
        if self.has_table(stripped):
            return False, "markdown table detected"
        lowered = stripped.lower()
        for marker in self.corruption_markers:
            if marker in lowered:
                return False, f"corruption marker detected: '{marker}'"
        if len(stripped) > 4000:
            return False, "draft too long / likely truncated"
        if stripped[-1] in "([{|":
            return False, "abrupt cutoff"
        return True, ""

    # ------------------------------------------------------------------
    # The Judge Revolver
    # ------------------------------------------------------------------

    def _judge_repair(
        self,
        ranked_chambers: List[Dict[str, Any]],
        draft: str,
        reason: str,
        custom_judge_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """
        Invoke the Judge: Ask a high-health chamber to repair a broken draft.
        """
        if not ranked_chambers:
            return None

        instruction_template = custom_judge_instruction or self.judge_instruction
        repair_instruction = instruction_template.format(reason=reason)

        judge_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": repair_instruction},
            {"role": "user", "content": f"Draft to repair:\n{draft}"},
        ]

        for judge in ranked_chambers[: self.judge_max_attempts]:
            log.info("Revolver Judge invoked on %s (reason: %s)", judge["name"], reason)
            content, outcome = self._call_chamber(judge, judge_messages)
            self._apply_outcome(judge, outcome)
            if content:
                cleaned = self.sanitize_draft(content)
                if cleaned:
                    log.info("Revolver Judge successfully repaired draft via %s", judge["name"])
                    return cleaned
        return None

    def _finalize_draft(
        self,
        ranked_chambers: List[Dict[str, Any]],
        draft: str,
        custom_judge_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Validate winning draft; mechanically sanitize or escalate to the Judge."""
        ok, reason = self.validate_draft(draft)
        if ok:
            return self.sanitize_draft(draft) or draft

        # Try mechanical sanitize first
        sanitized = self.sanitize_draft(draft)
        ok_s, _ = self.validate_draft(sanitized)
        if ok_s:
            return sanitized

        # Escalate to Judge
        repaired = self._judge_repair(ranked_chambers, sanitized or draft, reason, custom_judge_instruction)
        if repaired:
            ok2, _ = self.validate_draft(repaired)
            if ok2:
                return repaired
            repaired_clean = self.sanitize_draft(repaired)
            ok3, _ = self.validate_draft(repaired_clean)
            if ok3:
                return repaired_clean

        return None

    # ------------------------------------------------------------------
    # Main Public Execution Engine: ask()
    # ------------------------------------------------------------------

    def ask(
        self,
        user_message: str,
        extra_context: str = "",
        system_prompt: Optional[str] = None,
        judge_instruction: Optional[str] = None,
    ) -> str:
        """
        Process prompt through the Dual-Trigger, Self-Healing, Judge Revolver.

        1. Races top-2 healthy chambers concurrently.
        2. Falls back sequentially through remaining healthy chambers.
        3. Emergency fallback to cooled-down chambers if needed.
        4. Validates and auto-repairs outputs via the Judge.
        """
        if not self._chambers:
            return "⚠️ No AI chambers registered. Please add providers via add_provider() or add_chamber()."

        active_system = system_prompt or self.system_prompt
        messages = [{"role": "system", "content": active_system}]
        if extra_context:
            messages.append({"role": "system", "content": extra_context})
        messages.append({"role": "user", "content": user_message})

        deadline = time.monotonic() + self.timeout * 2
        healthy, cooled = self.rank_chambers(time.time())
        ranked = healthy + cooled

        # Tier 1: Dual-Trigger Race across top 2 healthy chambers
        racers = healthy[:2]
        if racers:
            content, winner_name = self._race(racers, messages, deadline)
            if content:
                final = self._finalize_draft(ranked, content, judge_instruction)
                if final:
                    return final

        # Tier 2: Sequential fallback through remaining healthy chambers
        for chamber in healthy[len(racers):]:
            if time.monotonic() >= deadline:
                break
            content, outcome = self._call_chamber(chamber, messages)
            self._apply_outcome(chamber, outcome)
            if content:
                final = self._finalize_draft(ranked, content, judge_instruction)
                if final:
                    return final

        # Tier 3: Emergency fallback through cooled chambers
        for chamber in cooled:
            if time.monotonic() >= deadline:
                break
            content, outcome = self._call_chamber(chamber, messages)
            self._apply_outcome(chamber, outcome)
            if content:
                final = self._finalize_draft(ranked, content, judge_instruction)
                if final:
                    return final

        return "🔌 All AI chambers are currently busy or unavailable. Please try again in a few moments."

    # ------------------------------------------------------------------
    # Telemetry & Status
    # ------------------------------------------------------------------

    def get_status(self) -> List[Dict[str, Any]]:
        """Return real-time health and cooldown status for all registered chambers."""
        now = time.time()
        results = []
        with self._health_lock:
            for chamber in self._chambers:
                name = chamber["name"]
                state = self._get_state_locked(name)
                self._decay_locked(state, now)
                is_cooldown = state["cooldown_until"] > now
                results.append({
                    "name": name,
                    "model": chamber["model"],
                    "health": round(state["health"], 1),
                    "status": "cooling" if is_cooldown else "healthy",
                    "cooldown_remaining": max(0.0, round(state["cooldown_until"] - now, 1)),
                    "successes": state["successes"],
                    "failures": state["failures"],
                })
        results.sort(key=lambda x: -x["health"])
        return results

    def close(self) -> None:
        """Close executor and HTTP session."""
        try:
            self._race_executor.shutdown(wait=False)
            self._session.close()
        except Exception as e:
            log.warning("Error during Revolver close: %s", e)


# ---------------------------------------------------------------------------
# Module-level Default Instance & Convenience Shortcuts
# ---------------------------------------------------------------------------
default_revolver = Revolver()

def add_chamber(*args, **kwargs):
    return default_revolver.add_chamber(*args, **kwargs)

def add_provider(*args, **kwargs):
    return default_revolver.add_provider(*args, **kwargs)

def ask(user_message: str, *args, **kwargs) -> str:
    return default_revolver.ask(user_message, *args, **kwargs)

def get_status() -> List[Dict[str, Any]]:
    return default_revolver.get_status()
