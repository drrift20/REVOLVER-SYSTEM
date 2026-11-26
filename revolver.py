"""
Revolver System — Self-Healing, Staggered-Hedged, Async & Sync Judge Revolver Engine.

A standalone, fault-tolerant AI orchestration library for staggered hedged racing,
adaptive health scoring, circuit breakers, and automated draft repair across both
synchronous (`Revolver`) and high-concurrency asynchronous (`AsyncRevolver`) runtimes.

Key Features:
- Zero hardcoded keys: Pass your own API keys for any OpenAI-compatible provider
  (Groq, Gemini, OpenRouter, OpenAI, Mistral, Cerebras, or custom endpoints).
- Dynamic Chamber Generation: Auto-discovers models directly from provider endpoints
  or accepts custom model lists to build chambers on the fly.
- Staggered Hedged Racing (Phase 1): Fires the healthiest chamber first and waits an
  adaptive delay (based on historical latency) before firing a backup racer.
  Eliminates P99 tail latency while saving ~70%+ redundant API quota/calls.
  Also supports immediate Dual-Trigger concurrent racing via `race_mode="concurrent"`.
- Native Async Engine (Phase 2): `AsyncRevolver` powered by `asyncio` + `httpx.AsyncClient`
  with connection limits and true coroutine cancellation of losing racers.
- Adaptive Health & Cooldowns: Per-chamber health tracking (0-100), latency EWMA,
  fast-response boosts, and granular penalties on 429, 5xx, timeouts, or model errors.
- The Judge Revolver: Conditional self-repair engine that detects malformed drafts
  (markdown tables, prompt leakage, cutoff text) and prompts a healthy chamber to fix it.
- Default English Judge + fully customizable judge instructions and system prompts.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Set, Tuple, Union

import requests
from requests.adapters import HTTPAdapter

try:
    import httpx
except ImportError:  # pragma: no cover - optional runtime dependency for AsyncRevolver
    httpx = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for Phase 5 JSON schema validation
    from pydantic import BaseModel
    from pydantic import ValidationError as PydanticValidationError
except ImportError:  # pragma: no cover
    BaseModel = None  # type: ignore[assignment, misc]
    PydanticValidationError = Exception  # type: ignore[assignment, misc]

log = logging.getLogger("revolver")

# ---------------------------------------------------------------------------
# Default Tuning Constants
# ---------------------------------------------------------------------------
DEFAULT_FAST_THRESHOLD = 4.0             # seconds: responses under this gain fast boost
DEFAULT_FAST_BOOST = 12.0                # health points added on fast success
DEFAULT_SLOW_BOOST = 6.0                 # health points added on slower success
DEFAULT_RECOVERY_PER_SEC = 100.0 / 240.0 # passive recovery: full 0->100 in ~4 minutes
DEFAULT_TIMEOUT_SECONDS = 15

# Phase 1: Staggered Hedged Racing Constants
DEFAULT_RACE_MODE = "hedged"             # "hedged" (staggered) or "concurrent" (immediate dual)
DEFAULT_HEDGE_DELAY = 0.75               # cold-start delay (s) before firing backup chamber
MIN_HEDGE_DELAY = 0.4                    # minimum adaptive wait before hedging (s)
MAX_HEDGE_DELAY = 2.5                    # maximum adaptive wait before hedging (s)
HEDGE_LATENCY_MULTIPLIER = 1.15          # hedge at 115% of primary chamber's avg latency
LATENCY_EWMA_ALPHA = 0.3                 # EWMA smoothing factor for chamber latency

# Phase 2: Async Connection Pool Limits
DEFAULT_ASYNC_MAX_KEEPALIVE = 20
DEFAULT_ASYNC_MAX_CONNECTIONS = 100

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

# Phase 5: JSON Schema Judge default instruction
DEFAULT_JSON_JUDGE_INSTRUCTION = (
    "The previous reply failed JSON schema validation with this error: {error}. "
    "Rewrite it as valid JSON that strictly conforms to the requested schema. "
    "Output ONLY the JSON object, with no markdown fences, no preamble, no explanation."
)

NO_CHAMBERS_MESSAGE = "⚠️ No AI chambers registered. Please add providers via add_provider() or add_chamber()."
BUSY_MESSAGE = "🔌 All AI chambers are currently busy or unavailable. Please try again in a few moments."


# ---------------------------------------------------------------------------
# Budget Guard — On-Device Paid API Spending Controls
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Raised when paid API budget is exhausted. Carries user-facing options."""
    def __init__(self, message: str, options: Optional[List[Dict[str, str]]] = None):
        self.options = options or []
        super().__init__(message)


class BudgetGuard:
    """
    Local, on-device budget guard for paid API chambers.

    Stores config in a JSON file next to the app. Never phones home.
    Supports monthly/yearly/custom cycles, warning thresholds, and
    ignore-once / ignore-forever overrides.
    """

    def __init__(self, config_path: str = "revolver_budget.json", auto_load: bool = True):
        self.config_path = config_path
        self.config: Dict[str, Any] = {
            "cycle_type": None,           # "monthly", "yearly", "custom"
            "custom_days": 30,
            "cycle_start": None,
            "limit_usd": None,
            "warning_threshold_pct": 80.0,
            "spent_usd": 0.0,
            "chambers": {},               # name -> {"paid": bool, "rate_prompt": float, "rate_completion": float}
            "ignore_forever": False,
            "ignore_until": None,
        }
        if auto_load:
            self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.config.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except OSError:
            pass

    # -- cycle logic --------------------------------------------------------

    def _cycle_expired(self) -> bool:
        if not self.config["cycle_start"] or not self.config["cycle_type"] or self.config["limit_usd"] is None:
            return True
        start = datetime.fromisoformat(self.config["cycle_start"])
        now = datetime.now()
        if self.config["cycle_type"] == "monthly":
            return (now - start).days >= 30
        if self.config["cycle_type"] == "yearly":
            return (now - start).days >= 365
        if self.config["cycle_type"] == "custom":
            return (now - start).days >= int(self.config.get("custom_days", 30))
        return True

    def _maybe_reset_cycle(self) -> None:
        if self._cycle_expired():
            self.config["spent_usd"] = 0.0
            self.config["cycle_start"] = datetime.now().isoformat()
            self.config["ignore_until"] = None
            self._save()

    # -- configuration ------------------------------------------------------

    def set_budget(
        self,
        limit_usd: float,
        cycle_type: str = "monthly",
        custom_days: int = 30,
        warning_threshold_pct: float = 80.0,
    ) -> None:
        """Set a spending cap. Cycle starts now."""
        self.config["cycle_type"] = cycle_type
        self.config["custom_days"] = custom_days
        self.config["cycle_start"] = datetime.now().isoformat()
        self.config["limit_usd"] = float(limit_usd)
        self.config["warning_threshold_pct"] = float(warning_threshold_pct)
        self.config["spent_usd"] = 0.0
        self.config["ignore_forever"] = False
        self.config["ignore_until"] = None
        self._save()

    def mark_paid(self, chamber_name: str, rate_prompt: float, rate_completion: float) -> None:
        """Tag a chamber as paid with per-1k-token rates (USD)."""
        self.config["chambers"][chamber_name] = {
            "paid": True,
            "rate_prompt": float(rate_prompt),
            "rate_completion": float(rate_completion),
        }
        self._save()

    def mark_free(self, chamber_name: str) -> None:
        """Tag a chamber as free (no budget deduction)."""
        self.config["chambers"][chamber_name] = {
            "paid": False,
            "rate_prompt": 0.0,
            "rate_completion": 0.0,
        }
        self._save()

    def is_paid(self, chamber_name: str) -> bool:
        chamber = self.config["chambers"].get(chamber_name)
        return bool(chamber.get("paid", False)) if chamber else False

    # -- enforcement --------------------------------------------------------

    def check(self, estimated_cost_usd: float = 0.0) -> None:
        """
        Raise BudgetExceeded if the hard limit is hit (and not ignored).
        Returns None if spending is allowed.
        """
        self._maybe_reset_cycle()

        if self.config["limit_usd"] is None:
            return  # No budget configured

        if self.config.get("ignore_forever"):
            return

        ignore_until = self.config.get("ignore_until")
        if ignore_until and datetime.now() < datetime.fromisoformat(ignore_until):
            self.config["ignore_until"] = None  # consume ignore-once after one pass
            self._save()
            return

        projected = self.config["spent_usd"] + estimated_cost_usd
        limit = self.config["limit_usd"]

        if projected >= limit:
            options = [
                {"label": "Raise budget", "action": "raise_budget"},
                {"label": "Ignore once", "action": "ignore_once"},
                {"label": "Ignore forever", "action": "ignore_forever"},
                {"label": "Auto-switch to free only", "action": "switch_free"},
            ]
            raise BudgetExceeded(
                f"Budget exhausted: ${self.config['spent_usd']:.4f} / ${limit:.4f} used",
                options=options,
            )

    def record_spend(self, amount_usd: float) -> None:
        self.config["spent_usd"] += float(amount_usd)
        self._save()

    def ignore_once(self) -> None:
        self.config["ignore_until"] = (datetime.now() + timedelta(hours=1)).isoformat()
        self.config["ignore_forever"] = False
        self._save()

    def ignore_forever(self) -> None:
        self.config["ignore_forever"] = True
        self.config["ignore_until"] = None
        self._save()

    def get_status(self) -> Dict[str, Any]:
        self._maybe_reset_cycle()
        limit = self.config["limit_usd"]
        spent = self.config["spent_usd"]
        return {
            "cycle_type": self.config["cycle_type"],
            "limit_usd": limit,
            "spent_usd": round(spent, 4),
            "remaining_usd": round(max(0.0, limit - spent), 4) if limit is not None else None,
            "warning_threshold_pct": self.config["warning_threshold_pct"],
            "ignore_forever": self.config.get("ignore_forever", False),
            "chambers": self.config["chambers"],
        }

# ---------------------------------------------------------------------------
# Phase 4: Complexity-Aware Model Tiering
# ---------------------------------------------------------------------------
TIER_SPRINT = "sprint"   # fast, cheap micro/flash models
TIER_HEAVY = "heavy"     # frontier / reasoning models
TIER_ANY = None          # chamber eligible for any tier

# Prompt complexity heuristic thresholds & markers
COMPLEXITY_LENGTH_THRESHOLD = 260          # chars above this lean toward "heavy"
COMPLEXITY_CODE_BLOCK_MARKERS = ("```", "~`~`~")
COMPLEXITY_REASONING_MARKERS = (
    "prove", "derive", "solve", "step by step", "step-by-step", "reasoning",
    "algorithm", "complex", "mathematical", "calculus", "theorem", "proof",
    "debug", "refactor", "architecture", "design pattern", "multi-step",
    "compare and contrast", "analyze", "explain in detail",
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
            "HTTP-Referer": "https://github.com/drrift20/REVOLVER-SYSTEM",
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
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "models_url": "https://api.deepseek.com/v1/models",
        "default_models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    },
    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "models_url": "https://api.together.xyz/v1/models",
        "default_models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ],
    },
    "fireworks": {
        "url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "models_url": "https://api.fireworks.ai/inference/v1/models",
        "default_models": [
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
        ],
    },
    "xai": {
        "url": "https://api.x.ai/v1/chat/completions",
        "models_url": "https://api.x.ai/v1/models",
        "default_models": [
            "grok-2-latest",
        ],
    },
    "ollama": {
        "url": "http://localhost:11434/v1/chat/completions",
        "models_url": "http://localhost:11434/v1/models",
        "default_models": [],
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "models_url": "https://api.anthropic.com/v1/models",
        "default_models": [
            "claude-3-5-sonnet-latest",
        ],
        "protocol": "anthropic",
        "extra_headers": {
            "anthropic-version": "2023-06-01",
        },
    },
}


# ---------------------------------------------------------------------------
# Universal Provider Auto-Detection, Dynamic Discovery & Dead-Model Pruning
# ---------------------------------------------------------------------------

# On-device persistence & cache locations (Linux + Windows via Path.home()).
REVOLVER_HOME = Path.home() / ".revolver"
MODELS_CACHE_PATH = REVOLVER_HOME / "models_cache.json"
CHAMBERS_STATE_PATH = REVOLVER_HOME / "chambers_state.json"

# Discovery cache time-to-live (seconds). Keeps startup/routing fast while
# avoiding rate-limit storms from re-fetching /models on every call.
DEFAULT_MODELS_CACHE_TTL = 6 * 60 * 60  # 6 hours

# Dead-model pruning defaults. Conservative so a brief 429 or single hiccup
# never wipes a good model.
DEFAULT_MAX_DOWN_DAYS = 3
DEFAULT_MIN_CONSECUTIVE_FAILURES = 8
DEFAULT_LOW_POOL_THRESHOLD = 1

# API-key prefix -> provider heuristics (checked before base_url inference).
API_KEY_PREFIX_HINTS: Tuple[Tuple[str, str], ...] = (
    ("gsk_", "groq"),
    ("AIza", "gemini"),
    ("sk-or-", "openrouter"),
    ("sk-ant-", "anthropic"),
    ("sk-proj-", "openai"),
    ("sk-", "openai"),
)

# base_url domain/path fragments -> provider heuristics.
BASE_URL_HINTS: Tuple[Tuple[str, str], ...] = (
    ("api.groq.com", "groq"),
    ("generativelanguage.googleapis.com", "gemini"),
    ("openrouter.ai", "openrouter"),
    ("api.openai.com", "openai"),
    ("api.anthropic.com", "anthropic"),
    ("api.deepseek.com", "deepseek"),
    ("api.together.xyz", "together"),
    ("api.fireworks.ai", "fireworks"),
    ("api.x.ai", "xai"),
    ("api.mistral.ai", "mistral"),
    ("api.cerebras.ai", "cerebras"),
    ("localhost:11434", "ollama"),
    ("127.0.0.1:11434", "ollama"),
)

# Model-name fragments that are NOT chat/completion models.
NON_CHAT_MODEL_MARKERS = (
    "embed", "whisper", "tts", "moderation", "dall-e", "audio",
    "rerank", "guard", "clip", "vision-encoder",
)

# Tier classification markers for discovered models.
SPRINT_MODEL_MARKERS = (
    "flash", "lite", "mini", "8b", "20b", "small", "fast", "nano", "tiny",
    "instant", "turbo",
)
HEAVY_MODEL_MARKERS = (
    "pro", "70b", "120b", "opus", "o1", "o3", "large", "max", "reasoner",
    "405b", "235b", "ultra",
)


def _ensure_revolver_home() -> None:
    """Best-effort creation of ~/.revolver (never raises)."""
    try:
        REVOLVER_HOME.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _infer_provider_from_key(api_key: Optional[str]) -> Optional[str]:
    """Infer a provider name from API-key prefix heuristics."""
    if not api_key:
        return None
    for prefix, provider in API_KEY_PREFIX_HINTS:
        if api_key.startswith(prefix):
            return provider
    return None


def _infer_provider_from_url(base_url: Optional[str]) -> Optional[str]:
    """Infer a provider name from a base_url domain/path."""
    if not base_url:
        return None
    lowered = base_url.lower()
    for fragment, provider in BASE_URL_HINTS:
        if fragment in lowered:
            return provider
    return None


def _chat_completions_url(base_url: str) -> str:
    """Derive a chat-completions endpoint from an arbitrary base_url."""
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1") or url.endswith("/v1beta") or url.endswith("/openai"):
        return f"{url}/chat/completions"
    if url.endswith("/messages"):
        return url
    return f"{url}/v1/chat/completions"


def _models_url(base_url: str) -> str:
    """Derive a /models endpoint from an arbitrary base_url."""
    url = base_url.rstrip("/")
    if url.endswith("/models"):
        return url
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")] + "/models"
    if url.endswith("/messages"):
        return url[: -len("/messages")] + "/models"
    if url.endswith("/v1") or url.endswith("/v1beta") or url.endswith("/openai"):
        return f"{url}/models"
    return f"{url}/v1/models"


def detect_provider(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Universal provider auto-detection.

    Resolution order:
      1. Explicit ``provider_hint`` (known name or custom name + base_url).
      2. ``base_url`` domain/path inference.
      3. API-key prefix heuristics.
      4. Clean fallback to an OpenAI-compatible adapter (custom base_url) or the
         ``openai`` provider when nothing else matches.

    Returns a normalized dict:
      {
        "provider": str,            # canonical/known name or custom slug
        "protocol": str,            # "openai" | "anthropic"
        "base_url": str,            # origin base url (may be "")
        "chat_url": str,            # chat-completions endpoint
        "models_url": str,          # /models endpoint (may be "")
        "headers":   dict,          # provider extra headers (no auth)
        "default_models": list,     # fallback model ids
        "known": bool,              # whether it matched KNOWN_PROVIDERS
      }
    """
    hint = (provider_hint or "").lower().strip()

    provider = None
    if hint:
        provider = hint
    if provider is None:
        provider = _infer_provider_from_url(base_url)
    if provider is None:
        provider = _infer_provider_from_key(api_key)
    if provider is None:
        # Fall back cleanly: custom base_url -> generic openai-compatible adapter,
        # otherwise assume openai.
        provider = "openai" if not base_url else "custom"

    info = KNOWN_PROVIDERS.get(provider, {})
    is_known = provider in KNOWN_PROVIDERS
    protocol = info.get("protocol", "openai")

    if base_url:
        origin = base_url.rstrip("/")
        chat_url = _chat_completions_url(base_url)
        models_u = _models_url(base_url)
        # Anthropic-style base urls keep the /messages protocol.
        if origin.endswith("/messages") or provider == "anthropic":
            protocol = "anthropic"
    else:
        origin = info.get("url", "")
        chat_url = info.get("url", "")
        models_u = info.get("models_url", "")

    headers = dict(info.get("extra_headers", {}))
    default_models = list(info.get("default_models", []))

    return {
        "provider": provider,
        "protocol": protocol,
        "base_url": origin,
        "chat_url": chat_url,
        "models_url": models_u,
        "headers": headers,
        "default_models": default_models,
        "known": is_known and not base_url,
    }


def classify_model_tier(model_id: str) -> Optional[str]:
    """
    Classify a discovered model id into a chamber tier.

    Returns ``TIER_SPRINT`` for fast/cheap models, ``TIER_HEAVY`` for
    frontier/reasoning models, or ``TIER_ANY`` (None) when ambiguous.
    """
    if not model_id:
        return TIER_ANY
    lowered = model_id.lower()
    if any(marker in lowered for marker in HEAVY_MODEL_MARKERS):
        return TIER_HEAVY
    if any(marker in lowered for marker in SPRINT_MODEL_MARKERS):
        return TIER_SPRINT
    return TIER_ANY


def _is_chat_capable_model(model_id: str) -> bool:
    """Return False for embedding/audio/moderation models that cannot chat."""
    lowered = model_id.lower()
    return not any(marker in lowered for marker in NON_CHAT_MODEL_MARKERS)


def _read_json_file(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk; return {} on any failure."""
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _write_json_file(path: Path, data: Dict[str, Any]) -> None:
    """Atomically-ish write a JSON object to disk; never raises."""
    try:
        _ensure_revolver_home()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except OSError:
        pass


def _load_models_cache() -> Dict[str, Any]:
    """Load the on-disk model-discovery cache (structure: {provider_key: entry})."""
    return _read_json_file(MODELS_CACHE_PATH)


def _save_models_cache(cache: Dict[str, Any]) -> None:
    _write_json_file(MODELS_CACHE_PATH, cache)


def _models_cache_key(provider: str, base_url: Optional[str]) -> str:
    return f"{provider}::{base_url or ''}"


def discover_provider_models(
    provider_name: str,
    api_key: str,
    base_url: Optional[str] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    max_models: int = 50,
    cache_ttl: int = DEFAULT_MODELS_CACHE_TTL,
) -> List[str]:
    """
    Discover chat-capable models for a provider with local caching.

    Fetches from the provider's ``/models`` endpoint (or a sensible fallback
    list when no endpoint is exposed), caches results in
    ``~/.revolver/models_cache.json`` with a TTL, and returns a filtered list of
    chat-capable model ids. ``force_refresh`` bypasses the cache; ``use_cache``
    disables reading/writing the cache entirely.
    """
    resolved = detect_provider(api_key=api_key, base_url=base_url, provider_hint=provider_name)
    models_url = resolved["models_url"]
    cache_key = _models_cache_key(resolved["provider"], base_url)

    if use_cache and not force_refresh:
        cache = _load_models_cache()
        entry = cache.get(cache_key)
        if isinstance(entry, dict):
            cached_at = entry.get("cached_at", 0)
            models = entry.get("models")
            if isinstance(models, list) and (time.time() - float(cached_at)) < cache_ttl:
                return models[:max_models]

    discovered: List[str] = []
    if models_url:
        headers = {"Authorization": f"Bearer {api_key}"}
        headers.update(resolved["headers"])
        try:
            resp = requests.get(models_url, headers=headers, timeout=8)
            if resp.status_code == 200:
                discovered = _filter_discovered_models(resp.json(), max_models=max_models)
            else:
                log.info("Model discovery for %s returned HTTP %s", provider_name, resp.status_code)
        except requests.exceptions.RequestException as exc:
            log.warning("Model discovery failed on %s: %s", models_url, exc)

    if not discovered:
        discovered = list(resolved["default_models"])[:max_models]

    if use_cache:
        cache = _load_models_cache()
        cache[cache_key] = {"cached_at": time.time(), "models": discovered}
        _save_models_cache(cache)

    return discovered


def _filter_discovered_models(data: Any, max_models: int = 50) -> List[str]:
    """Extract and filter chat-capable model ids from a /models JSON payload."""
    if not isinstance(data, dict):
        return []
    raw_models = data.get("data") or data.get("models") or []
    model_ids: List[str] = []
    for item in raw_models:
        m_id = item.get("id") or item.get("name") if isinstance(item, dict) else str(item)
        if m_id:
            if m_id.startswith("models/"):
                m_id = m_id.replace("models/", "")
            model_ids.append(m_id)
    if not model_ids:
        return []
    filtered = [m for m in model_ids if _is_chat_capable_model(m)]
    return (filtered or model_ids)[:max_models]


class ChamberStateStore:
    """
    Persistent per-chamber health & outage history (chambers_state.json).

    Tracks ``consecutive_failures``, ``last_ok_at``, ``first_down_at`` and a
    ``pruned`` flag per chamber/model so dead routes can be pruned across
    process restarts. Disk writes are throttled and best-effort; loading never
    auto-prunes live chambers (pruning is always an explicit action).
    """

    def __init__(self, path: Optional[str] = None, save_interval: float = 3.0):
        self.path = Path(path) if path else CHAMBERS_STATE_PATH
        self.save_interval = float(save_interval)
        self._lock = threading.Lock()
        self._last_save = 0.0
        self._dirty = False
        self.records: Dict[str, Dict[str, Any]] = _read_json_file(self.path)

    def record(self, name: str) -> Dict[str, Any]:
        """Fetch or create the persistent record for a chamber (caller holds lock)."""
        rec = self.records.get(name)
        if rec is None:
            rec = {
                "consecutive_failures": 0,
                "last_ok_at": None,
                "first_down_at": None,
                "last_failure_at": None,
                "pruned": False,
                "provider": None,
                "model": None,
            }
            self.records[name] = rec
        return rec

    def note_success(self, name: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            rec = self.record(name)
            rec["consecutive_failures"] = 0
            rec["last_ok_at"] = now
            rec["first_down_at"] = None
            rec["pruned"] = False
            self._dirty = True
        self._maybe_save()

    def note_failure(self, name: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            rec = self.record(name)
            rec["consecutive_failures"] = int(rec.get("consecutive_failures", 0)) + 1
            rec["last_failure_at"] = now
            if rec.get("first_down_at") is None:
                rec["first_down_at"] = now
            self._dirty = True
        self._maybe_save()

    def set_metadata(self, name: str, provider: Optional[str], model: Optional[str]) -> None:
        with self._lock:
            rec = self.record(name)
            if provider is not None:
                rec["provider"] = provider
            if model is not None:
                rec["model"] = model
            self._dirty = True

    def mark_pruned(self, name: str, pruned: bool = True) -> None:
        with self._lock:
            self.record(name)["pruned"] = bool(pruned)
            self._dirty = True
        self.save(force=True)

    def get(self, name: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self.records.get(name, {}))

    def all_records(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self.records.items()}

    def _maybe_save(self) -> None:
        if not self._dirty:
            return
        if (time.time() - self._last_save) < self.save_interval:
            return
        self.save()

    def save(self, force: bool = False) -> None:
        with self._lock:
            if not self._dirty and not force:
                return
            snapshot = {k: dict(v) for k, v in self.records.items()}
            self._dirty = False
            self._last_save = time.time()
        _write_json_file(self.path, snapshot)


class _BaseRevolver:
    """
    Shared core logic for synchronous `Revolver` and asynchronous `AsyncRevolver`.

    Encapsulates chamber registry, adaptive health & EWMA latency scoring,
    hedging math, HTTP response evaluation, text sanitization, and Judge logic.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful, concise, and intelligent AI assistant.",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        judge_max_attempts: int = 2,
        judge_instruction: Optional[str] = None,
        corruption_markers: Optional[Tuple[str, ...]] = None,
        race_mode: str = DEFAULT_RACE_MODE,
        hedge_delay: Optional[float] = None,
        min_hedge_delay: float = MIN_HEDGE_DELAY,
        max_hedge_delay: float = MAX_HEDGE_DELAY,
        hedge_multiplier: float = HEDGE_LATENCY_MULTIPLIER,
        cost_rates: Optional[Dict[str, Tuple[float, float]]] = None,
        budget_guard: Optional["BudgetGuard"] = None,
        state_path: Optional[str] = None,
        persist_state: bool = True,
    ):
        mode = self._validate_race_mode(race_mode)
        self.system_prompt = system_prompt
        self.timeout = float(timeout)
        self.judge_max_attempts = judge_max_attempts
        self.judge_instruction = judge_instruction or DEFAULT_JUDGE_INSTRUCTION
        self.corruption_markers = corruption_markers or DEFAULT_CORRUPTION_MARKERS

        self.race_mode = mode
        self.hedge_delay = float(hedge_delay) if hedge_delay is not None else None
        self.min_hedge_delay = max(0.0, float(min_hedge_delay))
        self.max_hedge_delay = max(self.min_hedge_delay, float(max_hedge_delay))
        self.hedge_multiplier = max(0.1, float(hedge_multiplier))

        # Phase 5: cost accounting. cost_rates[model] = (prompt_usd_per_1k, completion_usd_per_1k)
        self.cost_rates: Dict[str, Tuple[float, float]] = cost_rates or {}

        # Budget guard (on-device paid API spending controls)
        self._budget_guard = budget_guard

        # Persistent per-chamber outage/health history (dead-model pruning).
        self._state_store = ChamberStateStore(state_path) if persist_state else None

        # Thread safety & state storage (in-memory only; never held across I/O)
        self._health_lock = threading.Lock()
        self._chamber_state: Dict[str, Dict[str, Any]] = {}
        self._chambers: List[Dict[str, Any]] = []
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "single_chamber_calls": 0,
            "hedged_saved_calls": 0,
            "hedge_triggers": 0,
            "concurrent_races": 0,
            "judge_repairs": 0,
            "judge_failures": 0,
            "schema_repairs": 0,
            "schema_failures": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
        }

    @staticmethod
    def _validate_race_mode(race_mode: str) -> str:
        mode = (race_mode or DEFAULT_RACE_MODE).lower().strip()
        if mode not in ("hedged", "concurrent"):
            raise ValueError(f"Invalid race_mode '{race_mode}'. Expected 'hedged' or 'concurrent'.")
        return mode

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
        tier: Optional[str] = None,
        provider: Optional[str] = None,
        models_url: Optional[str] = None,
    ) -> None:
        """Register a single AI chamber and initialize its health state."""
        headers = {"Authorization": f"Bearer {api_key}"}
        if extra_headers:
            headers.update(extra_headers)

        chamber = {
            "name": name,
            "url": url,
            "key": api_key,
            "model": model,
            "headers": headers,
            "tier": tier,
            "provider": provider,
            "models_url": models_url,
            "pruned": False,
        }

        with self._health_lock:
            existing_idx = next((i for i, c in enumerate(self._chambers) if c["name"] == name), None)
            if existing_idx is not None:
                # Preserve pruned status across re-registration of a live name.
                chamber["pruned"] = self._chambers[existing_idx].get("pruned", False)
                self._chambers[existing_idx] = chamber
            else:
                self._chambers.append(chamber)
            self._get_state_locked(name)

        if self._state_store is not None:
            self._state_store.set_metadata(name, provider, model)

        # Register a safe free (0.0 rate) default with the budget guard so newly
        # discovered chambers never block free usage; explicit user overrides win.
        if self._budget_guard is not None:
            try:
                if name not in self._budget_guard.config.get("chambers", {}):
                    self._budget_guard.mark_free(name)
            except (OSError, ValueError):
                pass

        log.info("Registered chamber: %s (model: %s)", name, model)

    @staticmethod
    def _parse_discovered_models(data: Any, max_chambers: int = 5) -> List[str]:
        """Extract and filter chat-capable model IDs from a /models JSON payload."""
        if not isinstance(data, dict):
            return []
        raw_models = data.get("data") or data.get("models") or []
        model_ids: List[str] = []
        for item in raw_models:
            m_id = item.get("id") or item.get("name") if isinstance(item, dict) else str(item)
            if m_id:
                if m_id.startswith("models/"):
                    m_id = m_id.replace("models/", "")
                model_ids.append(m_id)
        if not model_ids:
            return []
        filtered = [
            m for m in model_ids
            if not any(
                skip in m.lower()
                for skip in ("embed", "whisper", "tts", "moderation", "dall-e", "audio")
            )
        ]
        return filtered[:max_chambers] if filtered else model_ids[:max_chambers]

    @staticmethod
    def _resolve_provider_config(
        provider: str,
        url: Optional[str],
        models_url: Optional[str],
        extra_headers: Optional[Dict[str, str]],
    ) -> Tuple[str, str, Optional[str], Dict[str, str], List[str]]:
        """Resolve URL, headers, and fallback default models for a named provider."""
        p_name = provider.lower().strip()
        provider_info = KNOWN_PROVIDERS.get(p_name, {})

        target_url = url or provider_info.get("url")
        if not target_url:
            raise ValueError(f"Unknown provider '{provider}' and no url provided.")

        m_url = models_url or provider_info.get("models_url")
        headers = provider_info.get("extra_headers", {}).copy()
        if extra_headers:
            headers.update(extra_headers)

        default_models = list(provider_info.get("default_models", []))
        return p_name, target_url, m_url, headers, default_models

    def _register_provider_models(
        self,
        p_name: str,
        target_url: str,
        api_key: str,
        headers: Dict[str, str],
        models: List[str],
        provider_label: str,
        tier: Optional[str] = None,
        models_url: Optional[str] = None,
    ) -> List[str]:
        """Register one chamber per resolved model name and return their chamber names."""
        if not models:
            raise ValueError(f"No models specified or discovered for provider '{provider_label}'.")

        created_chambers = []
        for model in models:
            chamber_name = f"{p_name.capitalize()} {model}"
            # Auto-classify tier per model when no explicit tier override is given.
            chamber_tier = tier if tier is not None else classify_model_tier(model)
            self.add_chamber(
                name=chamber_name,
                url=target_url,
                api_key=api_key,
                model=model,
                extra_headers=headers,
                tier=chamber_tier,
                provider=p_name,
                models_url=models_url,
            )
            created_chambers.append(chamber_name)
        return created_chambers

    # ------------------------------------------------------------------
    # Universal Discovery, Dead-Model Pruning & Healthy Replacement
    # ------------------------------------------------------------------

    def refresh_provider(
        self,
        provider: str,
        api_key: str,
        base_url: Optional[str] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
        max_models: int = 50,
        tier: Optional[str] = None,
    ) -> List[str]:
        """
        Discover a provider's models (cached) and register/refresh its chambers.

        Returns the list of chamber names now available for the provider.
        """
        resolved = detect_provider(api_key=api_key, base_url=base_url, provider_hint=provider)
        models = discover_provider_models(
            resolved["provider"],
            api_key,
            base_url=base_url,
            use_cache=use_cache,
            force_refresh=force_refresh,
            max_models=max_models,
        )
        if not models:
            models = list(resolved["default_models"])
        if not models:
            log.info("No models discovered for provider '%s'", provider)
            return []

        target_url = resolved["chat_url"] or resolved["base_url"]
        return self._register_provider_models(
            resolved["provider"],
            target_url,
            api_key,
            resolved["headers"],
            models,
            provider,
            tier=tier,
            models_url=resolved["models_url"],
        )

    def _active_chambers(self) -> List[Dict[str, Any]]:
        """Return chambers not currently pruned."""
        return [c for c in self._chambers if not c.get("pruned")]

    def prune_dead_models(
        self,
        max_down_days: int = DEFAULT_MAX_DOWN_DAYS,
        min_consecutive_failures: int = DEFAULT_MIN_CONSECUTIVE_FAILURES,
        force: bool = False,
        apply: bool = True,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Identify (and optionally mark) dead models/chambers as pruned.

        A chamber is pruned when it has been down longer than ``max_down_days``
        OR accumulated ``min_consecutive_failures`` consecutive failures.
        ``force`` prunes any chamber with any recorded outage regardless of
        thresholds. When ``apply`` is False this is a dry run.

        Returns a summary dict with ``pruned`` names and per-chamber ``events``.
        """
        now = now if now is not None else time.time()
        events: List[Dict[str, Any]] = []
        pruned_names: List[str] = []

        for chamber in list(self._chambers):
            if chamber.get("pruned"):
                continue
            name = chamber["name"]
            rec = self._state_store.get(name) if self._state_store is not None else {}
            consec = int(rec.get("consecutive_failures", 0) or 0)
            first_down = rec.get("first_down_at")
            down_days = ((now - float(first_down)) / 86400.0) if first_down else 0.0

            reason: Optional[str] = None
            if force:
                if consec > 0 or first_down is not None:
                    reason = "forced"
            else:
                if first_down is not None and down_days >= max_down_days:
                    reason = f"down {down_days:.1f}d >= {max_down_days}d"
                elif consec >= min_consecutive_failures:
                    reason = f"{consec} consecutive failures >= {min_consecutive_failures}"

            if reason is None:
                continue

            events.append({
                "name": name,
                "model": chamber.get("model"),
                "provider": chamber.get("provider"),
                "reason": reason,
                "consecutive_failures": consec,
                "down_days": round(down_days, 3),
            })
            pruned_names.append(name)

            if apply:
                chamber["pruned"] = True
                if self._state_store is not None:
                    self._state_store.mark_pruned(name, True)
                with self._health_lock:
                    st = self._get_state_locked(name)
                    st["health"] = 0.0
                    st["cooldown_until"] = now + 10 ** 9

        if apply and pruned_names:
            log.info("Pruned %d dead chamber(s): %s", len(pruned_names), pruned_names)
        else:
            log.info("prune_dead_models dry-run/scan: %d candidate(s)", len(pruned_names))

        return {"pruned": pruned_names, "events": events, "applied": apply}

    def replace_pruned_with_healthy(
        self,
        use_cache: bool = True,
        force_refresh: bool = False,
        max_models: int = 50,
        low_pool_threshold: int = DEFAULT_LOW_POOL_THRESHOLD,
        apply: bool = True,
    ) -> Dict[str, Any]:
        """
        Pull verified replacement models into chambers that were pruned or when the
        active pool drops low.

        Queries cached/fresh provider model discovery for each affected provider
        and registers healthy, non-duplicate models. Every discovery/prune/
        replacement event is logged. When ``apply`` is False this is a dry run.
        """
        events: List[Dict[str, Any]] = []
        added_names: List[str] = []

        pruned = [c for c in self._chambers if c.get("pruned")]
        active = self._active_chambers()

        # Providers to refresh: those with pruned chambers, plus (when the active
        # pool is low) every provider already represented in the roster.
        providers: Dict[str, Dict[str, Any]] = {}
        for c in pruned:
            p = c.get("provider")
            if p and p not in providers:
                providers[p] = c
        if len(active) <= low_pool_threshold:
            for c in self._chambers:
                p = c.get("provider")
                if p and p not in providers:
                    providers[p] = c

        existing_models = {c.get("model") for c in self._chambers}

        for provider, sample in providers.items():
            api_key = sample.get("key")
            if not api_key:
                continue
            base_url = sample.get("models_url") or sample.get("url")
            try:
                models = discover_provider_models(
                    provider,
                    api_key,
                    base_url=base_url,
                    use_cache=use_cache,
                    force_refresh=force_refresh,
                    max_models=max_models,
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Replacement discovery failed for '%s': %s", provider, exc)
                continue

            if not models:
                events.append({"provider": provider, "action": "no_models", "added": []})
                continue

            added_here: List[str] = []
            for model in models:
                if model in existing_models:
                    continue
                chamber_name = f"{provider.capitalize()} {model}"
                if any(c["name"] == chamber_name for c in self._chambers):
                    continue
                extra_headers = {
                    k: v for k, v in (sample.get("headers") or {}).items()
                    if k.lower() != "authorization"
                }
                if apply:
                    self.add_chamber(
                        name=chamber_name,
                        url=sample.get("url"),
                        api_key=api_key,
                        model=model,
                        extra_headers=extra_headers,
                        tier=classify_model_tier(model),
                        provider=provider,
                        models_url=sample.get("models_url"),
                    )
                existing_models.add(model)
                added_here.append(chamber_name)
                added_names.append(chamber_name)

            if added_here:
                log.info("Healthy replacement for '%s': +%s", provider, added_here)
            events.append({"provider": provider, "action": "replaced", "added": added_here})

        return {"added": added_names, "events": events, "applied": apply}

    def get_chamber_outage(self, name: str) -> Dict[str, Any]:
        """Return the persistent outage record for a chamber (empty if unknown)."""
        if self._state_store is None:
            return {}
        return self._state_store.get(name)

    # ------------------------------------------------------------------
    # Health Scoring, Latency EWMA & Chamber Ranking
    # ------------------------------------------------------------------

    def _get_state_locked(self, name: str) -> Dict[str, Any]:
        """Fetch or initialize a chamber's mutable health & latency record. Caller holds lock."""
        state = self._chamber_state.get(name)
        if state is None:
            state = {
                "health": 100.0,
                "cooldown_until": 0.0,
                "last_used": 0.0,
                "last_update": time.time(),
                "successes": 0,
                "failures": 0,
                "avg_latency": None,
                "last_latency": None,
                "latency_samples": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            }
            self._chamber_state[name] = state
        return state

    def _decay_locked(self, state: Dict[str, Any], now: float) -> None:
        """Heal a chamber's health back toward 100 based on elapsed time."""
        last = state["last_update"]
        if now > last:
            state["health"] = min(100.0, state["health"] + DEFAULT_RECOVERY_PER_SEC * (now - last))
            state["last_update"] = now

    def _apply_outcome(self, chamber: Dict[str, Any], outcome: Tuple[Any, ...]) -> None:
        """Fold request outcome into chamber's health, latency EWMA, tokens, and cooldown."""
        name = chamber["name"]
        kind = outcome[0]
        now = time.time()
        with self._health_lock:
            state = self._get_state_locked(name)
            self._decay_locked(state, now)
            if kind == "ok":
                elapsed = float(outcome[1]) if len(outcome) > 1 and outcome[1] is not None else 0.0
                boost = DEFAULT_FAST_BOOST if elapsed <= DEFAULT_FAST_THRESHOLD else DEFAULT_SLOW_BOOST
                state["health"] = min(100.0, state["health"] + boost)
                state["cooldown_until"] = 0.0
                state["last_used"] = now
                state["successes"] += 1

                state["last_latency"] = elapsed
                state["latency_samples"] += 1
                prev_avg = state["avg_latency"]
                if prev_avg is None:
                    state["avg_latency"] = elapsed
                else:
                    state["avg_latency"] = (
                        LATENCY_EWMA_ALPHA * elapsed + (1.0 - LATENCY_EWMA_ALPHA) * prev_avg
                    )

                if len(outcome) > 2 and isinstance(outcome[2], dict):
                    p_tok = int(outcome[2].get("prompt_tokens") or 0)
                    c_tok = int(outcome[2].get("completion_tokens") or 0)
                    state["prompt_tokens"] += p_tok
                    state["completion_tokens"] += c_tok
                    self._metrics["total_prompt_tokens"] += p_tok
                    self._metrics["total_completion_tokens"] += c_tok

                    # Phase 5: cost accounting per chamber model.
                    rate = self.cost_rates.get(chamber["model"])
                    if rate:
                        prompt_rate, completion_rate = rate
                        call_cost = (p_tok / 1000.0) * prompt_rate + (c_tok / 1000.0) * completion_rate
                        state["cost_usd"] += call_cost
                        self._metrics["total_cost_usd"] += call_cost
                        self._budget_record_spend(chamber, call_cost)
            else:
                penalty, cooldown = DEFAULT_OUTCOME_PENALTY.get(kind, (10.0, 8.0))
                state["health"] = max(0.0, state["health"] - penalty)
                state["cooldown_until"] = max(state["cooldown_until"], now + cooldown)
                state["failures"] += 1
            state["last_update"] = now

        # Persistent outage history for dead-model pruning (outside health lock).
        if self._state_store is not None:
            if kind == "ok":
                self._state_store.note_success(name, now)
            else:
                self._state_store.note_failure(name, now)

    def get_hedge_delay(
        self,
        primary_chamber: Dict[str, Any],
        override_delay: Optional[float] = None,
    ) -> float:
        """
        Compute the delay (in seconds) before firing a backup racer.

        Uses explicit override if given, instance `hedge_delay` if configured, or
        adapts dynamically from the primary chamber's EWMA latency.
        """
        if override_delay is not None:
            return max(0.0, float(override_delay))
        if self.hedge_delay is not None:
            return max(0.0, float(self.hedge_delay))

        with self._health_lock:
            state = self._get_state_locked(primary_chamber["name"])
            avg_lat = state.get("avg_latency")

        if avg_lat is None:
            return DEFAULT_HEDGE_DELAY

        adaptive = avg_lat * self.hedge_multiplier
        return max(self.min_hedge_delay, min(self.max_hedge_delay, adaptive))

    def rank_chambers(self, now: Optional[float] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (healthy, cooled) lists ranked by health, then least-recently-used."""
        if now is None:
            now = time.time()
        healthy = []
        cooled = []
        with self._health_lock:
            for chamber in self._chambers:
                if chamber.get("pruned"):
                    continue
                state = self._get_state_locked(chamber["name"])
                self._decay_locked(state, now)
                entry = (state["health"], state["last_used"], chamber)
                if state["cooldown_until"] > now:
                    cooled.append(entry)
                else:
                    healthy.append(entry)

        healthy.sort(key=lambda item: (-item[0], item[1]))
        cooled.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in healthy], [item[2] for item in cooled]

    @staticmethod
    def classify_prompt(text: str) -> str:
        """Heuristically classify a prompt as 'sprint' (simple) or 'heavy' (complex).

        Used by Phase 4 tiered routing: simple prompts race within sprint-tier chambers,
        complex prompts race within heavy-tier chambers.
        """
        if not text:
            return TIER_SPRINT
        lowered = text.lower()
        length = len(text)

        has_code = any(marker in text for marker in COMPLEXITY_CODE_BLOCK_MARKERS)
        has_reasoning = any(marker in lowered for marker in COMPLEXITY_REASONING_MARKERS)

        if has_code or has_reasoning or length > COMPLEXITY_LENGTH_THRESHOLD * 2:
            return TIER_HEAVY
        if length > COMPLEXITY_LENGTH_THRESHOLD or text.count("\n") > 6:
            return TIER_HEAVY if has_reasoning else TIER_SPRINT
        return TIER_SPRINT

    def _select_racers(
        self,
        healthy: List[Dict[str, Any]],
        cooled: List[Dict[str, Any]],
        classification: str,
    ) -> List[Dict[str, Any]]:
        """Pick up to two racer chambers, honoring tier tags when tiering is enabled."""
        tiered = [c for c in healthy + cooled if c.get("tier") is not None]
        if not tiered:
            return healthy[:2]

        def tier_matches(chamber: Dict[str, Any]) -> bool:
            tier = chamber.get("tier")
            if tier is None or tier == TIER_ANY:
                return True
            return tier == classification

        matched = [c for c in healthy + cooled if tier_matches(c)]
        if matched:
            return matched[:2]
        return (healthy + cooled)[:2]

    # ------------------------------------------------------------------
    # Request Preparation & Response Evaluation (Shared Sync/Async)
    # ------------------------------------------------------------------

    def _prepare_request(
        self,
        chamber: Dict[str, Any],
        messages: List[Dict[str, str]],
        timeout: Optional[float] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any], float]:
        """Build headers, JSON payload, and effective timeout for a chamber call."""
        headers = chamber["headers"].copy()
        headers["Content-Type"] = "application/json"
        payload = {
            "model": chamber["model"],
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.7,
        }
        req_timeout = timeout if timeout is not None else self.timeout
        return headers, payload, req_timeout

    @staticmethod
    def _evaluate_response(
        chamber: Dict[str, Any],
        resp: Any,
        elapsed: float,
    ) -> Tuple[Optional[str], Tuple[Any, ...]]:
        """Parse an HTTP response (requests or httpx) into (content_or_None, outcome_tuple)."""
        name = chamber["name"]
        status = resp.status_code

        if status == 200:
            try:
                data = resp.json()
                content = (data["choices"][0]["message"].get("content") or "").strip()
                usage = data.get("usage") if isinstance(data, dict) else None
            except (KeyError, IndexError, ValueError, AttributeError):
                log.warning("Chamber %s returned unexpected format", name)
                return None, ("parse", None)
            if content:
                return content, ("ok", elapsed, usage)
            return None, ("empty", None)

        if status == 429:
            log.warning("Chamber %s hit rate limit (429)", name)
            return None, ("429", None)
        if status in (400, 404):
            preview = (getattr(resp, "text", "") or "")[:140]
            log.warning("Chamber %s model error (%d): %s", name, status, preview)
            return None, ("model_error", status)
        if status >= 500:
            log.warning("Chamber %s failed with HTTP %d", name, status)
            return None, ("5xx", None)

        return None, ("httperr", status)

    def _build_messages(
        self,
        user_message: str,
        extra_context: str = "",
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Assemble system, optional context, and user messages."""
        active_system = system_prompt or self.system_prompt
        messages = [{"role": "system", "content": active_system}]
        if extra_context:
            messages.append({"role": "system", "content": extra_context})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _build_judge_messages(
        self,
        draft: str,
        reason: str,
        custom_judge_instruction: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build messages for The Judge repair call."""
        instruction_template = custom_judge_instruction or self.judge_instruction
        repair_instruction = instruction_template.format(reason=reason)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": repair_instruction},
            {"role": "user", "content": f"Draft to repair:\n{draft}"},
        ]

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

    @staticmethod
    def _parse_sse_delta(line: str) -> Optional[str]:
        """Extract a content delta from an OpenAI-compatible SSE streaming line."""
        if not line or not line.startswith("data:"):
            return None
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            return None
        try:
            delta = (data.get("choices") or [{}])[0].get("delta") or {}
            return delta.get("content") or ""
        except (AttributeError, IndexError, TypeError):
            return None

    @staticmethod
    def _parse_sse_usage(line: str) -> Optional[Dict[str, int]]:
        """Extract usage (prompt_tokens, completion_tokens) from an SSE data: line."""
        if not line or not line.startswith("data:"):
            return None
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            data = json.loads(payload)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        usage = data.get("usage")
        if isinstance(usage, dict):
            return {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
        return None

    def _try_sanitize_without_judge(self, draft: str) -> Tuple[Optional[str], str, str]:
        """Run Stages 1 & 2 validation/sanitization. Returns (valid_text_or_None, sanitized, reason)."""
        ok, reason = self.validate_draft(draft)
        if ok:
            return (self.sanitize_draft(draft) or draft), draft, ""

        sanitized = self.sanitize_draft(draft)
        ok_s, _ = self.validate_draft(sanitized)
        if ok_s:
            return sanitized, sanitized, ""

        return None, (sanitized or draft), reason

    def _accept_repaired_draft(self, repaired: Optional[str]) -> Optional[str]:
        """Validate Judge output and update judge metrics."""
        if repaired:
            ok2, _ = self.validate_draft(repaired)
            if ok2:
                with self._health_lock:
                    self._metrics["judge_repairs"] += 1
                return repaired
            repaired_clean = self.sanitize_draft(repaired)
            ok3, _ = self.validate_draft(repaired_clean)
            if ok3:
                with self._health_lock:
                    self._metrics["judge_repairs"] += 1
                return repaired_clean

        with self._health_lock:
            self._metrics["judge_failures"] += 1
        return None

    # ------------------------------------------------------------------
    # Phase 5: Structured JSON Schema Judge
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_text(content: str) -> str:
        """Strip leading/trailing markdown code fences (```json ... ``` or ``` ... ```) before JSON parse."""
        if not content:
            return content
        stripped = content.strip()
        # Remove opening fence
        if stripped.startswith("```"):
            # Find the end of the opening fence line
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            else:
                stripped = ""
        # Remove closing fence
        if stripped.rstrip().endswith("```"):
            last_newline = stripped.rfind("\n")
            if last_newline != -1:
                stripped = stripped[:last_newline]
            else:
                stripped = ""
        return stripped.strip()

    def _validate_schema(
        self,
        content: str,
        response_format: Any,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Parse content as JSON and validate against a Pydantic schema.

        Returns (parsed_object, None) on success, or (None, error_string) on failure.
        """
        if BaseModel is None or not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
            return None, "schema validation unavailable (pydantic not installed or response_format is not a Pydantic model)"
        if not content:
            return None, "empty response"
        try:
            data = json.loads(self._extract_json_text(content))
        except (ValueError, TypeError) as exc:
            return None, f"invalid JSON: {exc}"
        try:
            obj = response_format.model_validate(data)
        except PydanticValidationError as exc:
            return None, str(exc)
        return obj, None

    def _apply_schema_validation(
        self,
        ranked_chambers: List[Dict[str, Any]],
        draft: str,
        response_format: Any,
        custom_judge_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Repair a draft until it conforms to the Pydantic schema, using The Judge.

        Returns a valid JSON string conforming to the schema, or None if unrecoverable.
        """
        instruction_template = custom_judge_instruction or DEFAULT_JSON_JUDGE_INSTRUCTION
        current = draft
        for _ in range(max(1, self.judge_max_attempts)):
            obj, error = self._validate_schema(current, response_format)
            if obj is not None:
                with self._health_lock:
                    self._metrics["schema_repairs"] += 1
                return self._extract_json_text(current)  # already valid JSON, strip fences

            for judge in ranked_chambers[: self.judge_max_attempts]:
                log.info("Revolver Schema Judge invoked on %s (error: %s)", judge["name"], error)
                repair_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "system", "content": instruction_template.format(error=error)},
                    {"role": "user", "content": f"Invalid draft to repair:\n{current}"},
                ]
                content, outcome = self._call_chamber(judge, repair_messages)
                self._apply_outcome(judge, outcome)
                if content:
                    cleaned = self.sanitize_draft(content)
                    candidate = self._extract_json_text(cleaned or content)
                    obj2, _ = self._validate_schema(candidate, response_format)
                    if obj2 is not None:
                        with self._health_lock:
                            self._metrics["schema_repairs"] += 1
                        return candidate
                    current = candidate

        with self._health_lock:
            self._metrics["schema_failures"] += 1
        return None

    # ------------------------------------------------------------------
    # Telemetry & Status
    # ------------------------------------------------------------------

    def get_status(self) -> List[Dict[str, Any]]:
        """Return real-time health, latency, tokens, and cooldown status for all chambers."""
        now = time.time()
        results = []
        with self._health_lock:
            for chamber in self._chambers:
                name = chamber["name"]
                state = self._get_state_locked(name)
                self._decay_locked(state, now)
                is_cooldown = state["cooldown_until"] > now
                avg_lat = state["avg_latency"]
                last_lat = state["last_latency"]
                pruned = bool(chamber.get("pruned"))
                outage = self._state_store.get(name) if self._state_store is not None else {}
                if pruned:
                    status = "pruned"
                elif is_cooldown:
                    status = "cooling"
                else:
                    status = "healthy"
                results.append({
                    "name": name,
                    "model": chamber["model"],
                    "provider": chamber.get("provider"),
                    "tier": chamber.get("tier"),
                    "pruned": pruned,
                    "health": round(state["health"], 1),
                    "status": status,
                    "cooldown_remaining": max(0.0, round(state["cooldown_until"] - now, 1)),
                    "successes": state["successes"],
                    "failures": state["failures"],
                    "consecutive_failures": int(outage.get("consecutive_failures", 0) or 0),
                    "first_down_at": outage.get("first_down_at"),
                    "last_ok_at": outage.get("last_ok_at"),
                    "avg_latency_ms": round(avg_lat * 1000.0, 1) if avg_lat is not None else None,
                    "last_latency_ms": round(last_lat * 1000.0, 1) if last_lat is not None else None,
                    "prompt_tokens": state["prompt_tokens"],
                    "completion_tokens": state["completion_tokens"],
                    "cost_usd": round(state["cost_usd"], 6),
                })
        results.sort(key=lambda x: -x["health"])
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregate engine telemetry, including hedged quota savings."""
        with self._health_lock:
            m = self._metrics.copy()

        hedged_total = m["hedged_saved_calls"] + m["hedge_triggers"]
        m["race_mode"] = self.race_mode
        m["hedge_efficiency_pct"] = (
            round((m["hedged_saved_calls"] / hedged_total) * 100.0, 1)
            if hedged_total > 0
            else 0.0
        )
        m["calls_saved"] = m["hedged_saved_calls"]
        return m

    def export_prometheus(self) -> str:
        """Export chamber & engine telemetry in Prometheus/OpenTelemetry text format.

        Metrics:
          revolver_requests_total
          revolver_requests_saved_total (hedged backup calls avoided)
          revolver_chamber_health{chamber,model,tier}
          revolver_chamber_cost_usd_total{chamber,model}
          revolver_chamber_prompt_tokens_total{chamber,model}
          revolver_chamber_completion_tokens_total{chamber,model}
          revolver_chamber_successes_total{chamber,model}
          revolver_chamber_failures_total{chamber,model}
          revolver_engine_cost_usd_total
          revolver_engine_prompt_tokens_total
          revolver_engine_completion_tokens_total
        """
        now = time.time()
        lines: List[str] = []
        with self._health_lock:
            m = self._metrics.copy()
            chambers = list(self._chambers)
            states = {c["name"]: self._get_state_locked(c["name"]) for c in chambers}

        def esc(label: str) -> str:
            return str(label).replace("\\", "\\\\").replace('"', '\\"') if label is not None else ""

        lines.append('# HELP revolver_requests_total Total requests processed by the engine')
        lines.append('# TYPE revolver_requests_total counter')
        lines.append(f'revolver_requests_total {m["total_requests"]}')
        lines.append('# HELP revolver_requests_saved_total Hedged backup API calls avoided')
        lines.append('# TYPE revolver_requests_saved_total counter')
        lines.append(f'revolver_requests_saved_total {m["hedged_saved_calls"]}')
        lines.append('# HELP revolver_engine_cost_usd_total Estimated total spend in USD')
        lines.append('# TYPE revolver_engine_cost_usd_total counter')
        lines.append(f'revolver_engine_cost_usd_total {m.get("total_cost_usd", 0.0)}')
        lines.append('# HELP revolver_engine_prompt_tokens_total Total prompt tokens')
        lines.append('# TYPE revolver_engine_prompt_tokens_total counter')
        lines.append(f'revolver_engine_prompt_tokens_total {m["total_prompt_tokens"]}')
        lines.append('# HELP revolver_engine_completion_tokens_total Total completion tokens')
        lines.append('# TYPE revolver_engine_completion_tokens_total counter')
        lines.append(f'revolver_engine_completion_tokens_total {m["total_completion_tokens"]}')

        for chamber in chambers:
            name = esc(chamber["name"])
            model = esc(chamber["model"])
            tier = esc(chamber.get("tier") or "any")
            state = states[chamber["name"]]
            self._decay_locked(state, now)
            lines.append('# HELP revolver_chamber_health Chamber health score (0-100)')
            lines.append('# TYPE revolver_chamber_health gauge')
            lines.append(f'revolver_chamber_health{{chamber="{name}",model="{model}",tier="{tier}"}} {state["health"]}')
            lines.append('# HELP revolver_chamber_cost_usd_total Estimated chamber spend in USD')
            lines.append('# TYPE revolver_chamber_cost_usd_total counter')
            lines.append(f'revolver_chamber_cost_usd_total{{chamber="{name}",model="{model}"}} {state.get("cost_usd", 0.0)}')
            lines.append('# HELP revolver_chamber_prompt_tokens_total Chamber prompt tokens')
            lines.append('# TYPE revolver_chamber_prompt_tokens_total counter')
            lines.append(f'revolver_chamber_prompt_tokens_total{{chamber="{name}",model="{model}"}} {state["prompt_tokens"]}')
            lines.append('# HELP revolver_chamber_completion_tokens_total Chamber completion tokens')
            lines.append('# TYPE revolver_chamber_completion_tokens_total counter')
            lines.append(f'revolver_chamber_completion_tokens_total{{chamber="{name}",model="{model}"}} {state["completion_tokens"]}')
            lines.append('# HELP revolver_chamber_successes_total Chamber successful requests')
            lines.append('# TYPE revolver_chamber_successes_total counter')
            lines.append(f'revolver_chamber_successes_total{{chamber="{name}",model="{model}"}} {state["successes"]}')
            lines.append('# HELP revolver_chamber_failures_total Chamber failed requests')
            lines.append('# TYPE revolver_chamber_failures_total counter')
            lines.append(f'revolver_chamber_failures_total{{chamber="{name}",model="{model}"}} {state["failures"]}')

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Budget Guard API
    # ------------------------------------------------------------------

    def set_budget(
        self,
        limit_usd: float,
        cycle_type: str = "monthly",
        custom_days: int = 30,
        warning_threshold_pct: float = 80.0,
        config_path: Optional[str] = None,
    ) -> None:
        """Configure on-device spending cap for paid chambers."""
        if self._budget_guard is None:
            self._budget_guard = BudgetGuard(config_path=config_path or "revolver_budget.json")
        self._budget_guard.set_budget(
            limit_usd=limit_usd,
            cycle_type=cycle_type,
            custom_days=custom_days,
            warning_threshold_pct=warning_threshold_pct,
        )

    def mark_chamber_paid(self, name: str, rate_prompt: float, rate_completion: float) -> None:
        """Tag a chamber as paid with per-1k-token rates (USD)."""
        if self._budget_guard is None:
            self._budget_guard = BudgetGuard()
        self._budget_guard.mark_paid(name, rate_prompt, rate_completion)

    def mark_chamber_free(self, name: str) -> None:
        """Tag a chamber as free (no budget deduction)."""
        if self._budget_guard is None:
            self._budget_guard = BudgetGuard()
        self._budget_guard.mark_free(name)

    def get_budget_status(self) -> Dict[str, Any]:
        """Return current budget status (or empty dict if no guard configured)."""
        if self._budget_guard is None:
            return {"budget_configured": False}
        return {"budget_configured": True, **self._budget_guard.get_status()}

    def _budget_check(self, estimated_cost_usd: float = 0.0) -> None:
        """Raise BudgetExceeded if paid spending is blocked."""
        if self._budget_guard is not None:
            self._budget_guard.check(estimated_cost_usd)

    def _budget_record_spend(self, chamber: Dict[str, Any], amount_usd: float) -> None:
        """Record spend if chamber is paid and guard is configured."""
        if self._budget_guard is not None and self._budget_guard.is_paid(chamber["name"]):
            self._budget_guard.record_spend(amount_usd)


# ===========================================================================
# Synchronous Engine (`Revolver`)
# ===========================================================================

class Revolver(_BaseRevolver):
    """
    Synchronous Revolver Engine powered by pooled `requests.Session` & `ThreadPoolExecutor`.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful, concise, and intelligent AI assistant.",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_race_workers: int = 4,
        judge_max_attempts: int = 2,
        judge_instruction: Optional[str] = None,
        corruption_markers: Optional[Tuple[str, ...]] = None,
        session_pool_size: int = 8,
        race_mode: str = DEFAULT_RACE_MODE,
        hedge_delay: Optional[float] = None,
        min_hedge_delay: float = MIN_HEDGE_DELAY,
        max_hedge_delay: float = MAX_HEDGE_DELAY,
        hedge_multiplier: float = HEDGE_LATENCY_MULTIPLIER,
        cost_rates: Optional[Dict[str, Tuple[float, float]]] = None,
        budget_guard: Optional["BudgetGuard"] = None,
        state_path: Optional[str] = None,
        persist_state: bool = True,
    ):
        super().__init__(
            system_prompt=system_prompt,
            timeout=timeout,
            judge_max_attempts=judge_max_attempts,
            judge_instruction=judge_instruction,
            corruption_markers=corruption_markers,
            race_mode=race_mode,
            hedge_delay=hedge_delay,
            min_hedge_delay=min_hedge_delay,
            max_hedge_delay=max_hedge_delay,
            hedge_multiplier=hedge_multiplier,
            cost_rates=cost_rates,
            budget_guard=budget_guard,
            state_path=state_path,
            persist_state=persist_state,
        )
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=session_pool_size)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        self._race_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_race_workers, thread_name_prefix="revolver-race"
        )

    def __enter__(self) -> "Revolver":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def discover_models(self, models_url: str, api_key: str) -> List[str]:
        """Query provider's /models endpoint synchronously and return normalized model IDs."""
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            resp = self._session.get(models_url, headers=headers, timeout=8)
            if resp.status_code == 200:
                return self._parse_discovered_models(resp.json(), max_chambers=50)
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
        tier: Optional[str] = None,
    ) -> List[str]:
        """Register a provider synchronously, optionally auto-discovering live models."""
        p_name, target_url, m_url, headers, default_models = self._resolve_provider_config(
            provider, url, models_url, extra_headers
        )

        target_models: List[str] = list(models) if models else []
        if not target_models and discover_models and m_url:
            headers_auth = {"Authorization": f"Bearer {api_key}"}
            try:
                resp = self._session.get(m_url, headers=headers_auth, timeout=8)
                if resp.status_code == 200:
                    target_models = self._parse_discovered_models(resp.json(), max_chambers=max_chambers)
            except Exception as exc:
                log.warning("Model discovery failed on %s: %s", m_url, exc)

        if not target_models:
            target_models = default_models

        return self._register_provider_models(
            p_name, target_url, api_key, headers, target_models, provider, tier=tier,
            models_url=m_url,
        )

    def _call_chamber(
        self,
        chamber: Dict[str, Any],
        messages: List[Dict[str, str]],
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[str], Tuple[Any, ...]]:
        """Fire a single synchronous chamber request."""
        headers, payload, req_timeout = self._prepare_request(chamber, messages, timeout)
        started = time.monotonic()
        try:
            resp = self._session.post(chamber["url"], headers=headers, json=payload, timeout=req_timeout)
        except requests.exceptions.Timeout:
            return None, ("timeout", None)
        except requests.exceptions.RequestException as exc:
            log.warning("Chamber %s network exception: %s", chamber["name"], exc)
            return None, ("network", None)

        elapsed = time.monotonic() - started
        return self._evaluate_response(chamber, resp, elapsed)

    def _attach_loser_callback(
        self,
        future: concurrent.futures.Future,
        chamber: Dict[str, Any],
        processed: Set[concurrent.futures.Future],
    ) -> None:
        """Cancel a slow competitor if queued; otherwise record its late outcome for EWMA/health."""
        if future in processed:
            return
        if future.cancel():
            processed.add(future)
            return

        processed.add(future)

        def _on_done(fut: concurrent.futures.Future) -> None:
            if fut.cancelled():
                return
            try:
                _, outcome = fut.result()
            except Exception:
                outcome = ("exception", None)
            self._apply_outcome(chamber, outcome)

        future.add_done_callback(_on_done)

    def _await_futures_winner(
        self,
        future_map: Dict[concurrent.futures.Future, Dict[str, Any]],
        deadline: float,
        processed: Optional[Set[concurrent.futures.Future]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Wait for first future returning valid non-empty content; cancel/detach losers."""
        if processed is None:
            processed = set()
        winner_content = None
        winner_name = None
        remaining = max(0.1, min(deadline - time.monotonic(), self.timeout + 2.0))
        try:
            for future in concurrent.futures.as_completed(future_map, timeout=remaining):
                chamber = future_map[future]
                if future not in processed:
                    processed.add(future)
                    try:
                        content, outcome = future.result()
                    except Exception as exc:
                        log.warning("Racer %s raised: %s", chamber["name"], exc)
                        content, outcome = None, ("exception", None)
                    self._apply_outcome(chamber, outcome)
                else:
                    content = None

                if content:
                    winner_content = content
                    winner_name = chamber["name"]
                    for other, other_chamber in future_map.items():
                        if other is not future:
                            self._attach_loser_callback(other, other_chamber, processed)
                    break
        except concurrent.futures.TimeoutError:
            log.warning("Revolver race timed out across %d chamber(s)", len(future_map))
            for other, other_chamber in future_map.items():
                self._attach_loser_callback(other, other_chamber, processed)

        return winner_content, winner_name

    def _race_concurrent(
        self,
        chambers: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        deadline: float,
    ) -> Tuple[Optional[str], Optional[str], Set[str]]:
        """Immediate Dual-Trigger race: fire all candidate chambers simultaneously."""
        attempted: Set[str] = set()
        if not chambers:
            return None, None, attempted

        if len(chambers) == 1:
            attempted.add(chambers[0]["name"])
            req_timeout = max(0.1, min(self.timeout, deadline - time.monotonic()))
            content, outcome = self._call_chamber(chambers[0], messages, timeout=req_timeout)
            self._apply_outcome(chambers[0], outcome)
            with self._health_lock:
                self._metrics["single_chamber_calls"] += 1
            return content, (chambers[0]["name"] if content else None), attempted

        with self._health_lock:
            self._metrics["concurrent_races"] += 1

        req_timeout = max(0.1, min(self.timeout, deadline - time.monotonic()))
        future_map = {}
        for chamber in chambers:
            attempted.add(chamber["name"])
            fut = self._race_executor.submit(self._call_chamber, chamber, messages, req_timeout)
            future_map[fut] = chamber

        winner_content, winner_name = self._await_futures_winner(future_map, deadline)
        return winner_content, winner_name, attempted

    def _race_hedged(
        self,
        chambers: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        deadline: float,
        override_hedge_delay: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[str], Set[str]]:
        """Staggered Hedged Race across top-2 healthy chambers."""
        attempted: Set[str] = set()
        if not chambers:
            return None, None, attempted
        if len(chambers) == 1:
            return self._race_concurrent(chambers, messages, deadline)

        c1, c2 = chambers[0], chambers[1]
        delta_t = self.get_hedge_delay(c1, override_delay=override_hedge_delay)
        if delta_t <= 0.0:
            return self._race_concurrent([c1, c2], messages, deadline)

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None, None, attempted

        attempted.add(c1["name"])
        req_timeout_1 = max(0.1, min(self.timeout, remaining))
        f1 = self._race_executor.submit(self._call_chamber, c1, messages, req_timeout_1)

        wait_budget = min(delta_t, max(0.05, remaining))
        done, _ = concurrent.futures.wait([f1], timeout=wait_budget)

        if f1 in done:
            try:
                content1, outcome1 = f1.result()
            except Exception as exc:
                log.warning("Primary racer %s raised: %s", c1["name"], exc)
                content1, outcome1 = None, ("exception", None)
            self._apply_outcome(c1, outcome1)

            if content1:
                with self._health_lock:
                    self._metrics["hedged_saved_calls"] += 1
                log.info("Hedged win (saved backup call): %s", c1["name"])
                return content1, c1["name"], attempted

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None, None, attempted

            attempted.add(c2["name"])
            with self._health_lock:
                self._metrics["hedge_triggers"] += 1
            log.info(
                "Primary %s failed fast (%s); firing backup %s immediately",
                c1["name"], outcome1[0], c2["name"],
            )
            req_timeout_2 = max(0.1, min(self.timeout, remaining))
            f2 = self._race_executor.submit(self._call_chamber, c2, messages, req_timeout_2)
            winner_content, winner_name = self._await_futures_winner({f2: c2}, deadline)
            return winner_content, winner_name, attempted

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            self._attach_loser_callback(f1, c1, set())
            return None, None, attempted

        attempted.add(c2["name"])
        with self._health_lock:
            self._metrics["hedge_triggers"] += 1
        log.info("Hedge triggered after %.2fs: firing %s alongside %s", delta_t, c2["name"], c1["name"])

        req_timeout_2 = max(0.1, min(self.timeout, remaining))
        f2 = self._race_executor.submit(self._call_chamber, c2, messages, req_timeout_2)
        winner_content, winner_name = self._await_futures_winner({f1: c1, f2: c2}, deadline)
        return winner_content, winner_name, attempted

    def _race(
        self,
        chambers: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        deadline: float,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Backwards-compatible concurrent race alias."""
        content, winner, _ = self._race_concurrent(chambers, messages, deadline)
        return content, winner

    def _judge_repair(
        self,
        ranked_chambers: List[Dict[str, Any]],
        draft: str,
        reason: str,
        custom_judge_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Invoke the Judge synchronously on up to `judge_max_attempts` chambers."""
        if not ranked_chambers:
            return None

        judge_messages = self._build_judge_messages(draft, reason, custom_judge_instruction)
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
        response_format: Any = None,
    ) -> Optional[str]:
        """Validate winning draft; mechanically sanitize or escalate synchronously to the Judge.

        When ``response_format`` is a Pydantic model class, the draft is also validated
        against the JSON schema and auto-repaired by The Schema Judge when invalid.
        """
        clean_or_none, fallback_draft, reason = self._try_sanitize_without_judge(draft)
        if clean_or_none is not None:
            candidate = clean_or_none
        else:
            repaired = self._judge_repair(ranked_chambers, fallback_draft, reason, custom_judge_instruction)
            candidate = self._accept_repaired_draft(repaired)
            if candidate is None:
                return None

        if response_format is not None:
            return self._apply_schema_validation(ranked_chambers, candidate, response_format, custom_judge_instruction)
        return candidate

    def ask(
        self,
        user_message: str,
        extra_context: str = "",
        system_prompt: Optional[str] = None,
        judge_instruction: Optional[str] = None,
        race_mode: Optional[str] = None,
        hedge_delay: Optional[float] = None,
        stream: bool = False,
        response_format: Any = None,
    ) -> Union[str, Generator[str, None, None]]:
        """Process prompt synchronously through Staggered Hedging / Concurrent Race + Judge.

        When ``stream=True``, returns a generator yielding content chunks live (TTFT race)
        instead of a single completed string. The Judge repair loop is skipped for streaming.
        When ``response_format`` is a Pydantic model class, the reply is validated against
        its JSON schema and auto-healed by The Schema Judge.
        """
        if stream:
            return self._ask_streaming(
                user_message,
                extra_context=extra_context,
                system_prompt=system_prompt,
                judge_instruction=judge_instruction,
                race_mode=race_mode,
                hedge_delay=hedge_delay,
            )
        if not self._chambers:
            return NO_CHAMBERS_MESSAGE

        active_mode = self._validate_race_mode(race_mode or self.race_mode)
        with self._health_lock:
            self._metrics["total_requests"] += 1

        messages = self._build_messages(user_message, extra_context, system_prompt)
        deadline = time.monotonic() + self.timeout * 2
        healthy, cooled = self.rank_chambers(time.time())
        ranked = healthy + cooled

        classification = self.classify_prompt(user_message)
        attempted_names: Set[str] = set()
        racers = self._select_racers(healthy, cooled, classification)

        # Budget guard: if paid racers are blocked, fall back to free-only chambers.
        budget_exhausted = False
        if racers and self._budget_guard is not None:
            paid_racers = [c for c in racers if self._budget_guard.is_paid(c["name"])]
            if paid_racers:
                try:
                    self._budget_check()
                except BudgetExceeded:
                    log.warning("Budget exhausted; skipping paid chambers, falling back to free-only")
                    budget_exhausted = True
                    racers = [c for c in racers if not self._budget_guard.is_paid(c["name"])]
                    if not racers:
                        return BUSY_MESSAGE + " (budget exhausted, no free chambers available)"

        if racers:
            if active_mode == "concurrent":
                content, _, attempted_names = self._race_concurrent(racers, messages, deadline)
            else:
                content, _, attempted_names = self._race_hedged(
                    racers, messages, deadline, override_hedge_delay=hedge_delay
                )
            if content:
                final = self._finalize_draft(ranked, content, judge_instruction, response_format=response_format)
                if final:
                    return final

        for chamber in healthy + cooled:
            if chamber["name"] in attempted_names:
                continue
            # Budget guard: skip paid chambers if budget is exhausted
            if self._budget_guard is not None and self._budget_guard.is_paid(chamber["name"]):
                if budget_exhausted:
                    continue
                try:
                    self._budget_check()
                except BudgetExceeded:
                    log.warning("Budget exhausted; skipping paid chamber %s in fallback loop", chamber["name"])
                    budget_exhausted = True
                    continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            req_timeout = max(0.1, min(self.timeout, remaining))
            content, outcome = self._call_chamber(chamber, messages, timeout=req_timeout)
            self._apply_outcome(chamber, outcome)
            attempted_names.add(chamber["name"])
            if content:
                final = self._finalize_draft(ranked, content, judge_instruction, response_format=response_format)
                if final:
                    return final

        return BUSY_MESSAGE

    def _ask_streaming(
        self,
        user_message: str,
        extra_context: str = "",
        system_prompt: Optional[str] = None,
        judge_instruction: Optional[str] = None,
        race_mode: Optional[str] = None,
        hedge_delay: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """Stream a reply via TTFT race: first chamber to yield a chunk wins; losers are cut off."""
        if not self._chambers:
            yield NO_CHAMBERS_MESSAGE
            return

        active_mode = self._validate_race_mode(race_mode or self.race_mode)
        del active_mode  # streaming always fires both racers concurrently for a true TTFT race
        with self._health_lock:
            self._metrics["total_requests"] += 1

        messages = self._build_messages(user_message, extra_context, system_prompt)
        deadline = time.monotonic() + self.timeout * 2
        healthy, cooled = self.rank_chambers(time.time())
        classification = self.classify_prompt(user_message)
        racers = self._select_racers(healthy, cooled, classification)
        racers_map = {c["name"]: c for c in racers}

        result_q: "queue.Queue[Tuple[str, ...]]" = queue.Queue()
        stop_events: Dict[str, threading.Event] = {}
        threads: Dict[str, threading.Thread] = {}

        def _run(chamber: Dict[str, Any]) -> None:
            name = chamber["name"]
            stop = stop_events[name]
            headers, payload, req_timeout = self._prepare_request(chamber, messages)
            headers["Accept"] = "text/event-stream"
            payload["stream"] = True
            resp = None
            try:
                resp = self._session.post(
                    chamber["url"], headers=headers, json=payload, stream=True, timeout=req_timeout
                )
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if stop.is_set():
                        break
                    if not raw_line:
                        continue
                    chunk = self._parse_sse_delta(raw_line)
                    usage = self._parse_sse_usage(raw_line)
                    if chunk:
                        result_q.put(("chunk", name, chunk))
                    if usage:
                        result_q.put(("usage", name, usage))
            except Exception as exc:  # noqa: BLE001 - stream failures are reported as chunks/errors
                log.warning("Streaming chamber %s errored: %s", name, exc)
                result_q.put(("error", name, str(exc)))
            finally:
                result_q.put(("done", name))
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:  # noqa: BLE001
                        pass

        for chamber in racers:
            stop_events[chamber["name"]] = threading.Event()
            t = threading.Thread(target=_run, args=(chamber,), daemon=True)
            threads[chamber["name"]] = t
            t.start()

        winner: Optional[str] = None
        finished: Set[str] = set()
        yielded_any = False
        started = time.monotonic()
        winner_usage: Optional[Dict[str, int]] = None
        try:
            while len(finished) < len(racers):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    log.warning("Streaming race reached deadline across %d chamber(s)", len(racers))
                    break
                try:
                    item = result_q.get(timeout=min(0.2, remaining))
                except queue.Empty:
                    continue
                kind = item[0]
                if kind == "done":
                    finished.add(item[1])
                    continue
                if kind == "error":
                    finished.add(item[1])
                    continue
                if kind == "usage":
                    _, name, usage = item
                    if winner is not None and name == winner:
                        winner_usage = usage
                    continue
                if kind == "chunk":
                    _, name, chunk = item
                    if winner is None:
                        winner = name
                        yielded_any = True
                        with self._health_lock:
                            self._metrics["concurrent_races"] += 1
                        for other_name, ev in stop_events.items():
                            if other_name != name:
                                ev.set()
                    if winner == name:
                        yield chunk
                        yielded_any = True
        finally:
            for ev in stop_events.values():
                ev.set()

        # Best-effort health accounting for the streaming race.
        elapsed = time.monotonic() - started
        if winner is not None and yielded_any:
            self._apply_outcome(racers_map[winner], ("ok", elapsed, winner_usage))
        elif racers:
            for chamber in racers:
                if chamber["name"] != winner:
                    self._apply_outcome(chamber, ("network", None))

    def close(self) -> None:
        """Close thread pool executor and HTTP session."""
        try:
            self._race_executor.shutdown(wait=False)
            self._session.close()
        except Exception as e:
            log.warning("Error during Revolver close: %s", e)
        finally:
            if self._state_store is not None:
                try:
                    self._state_store.save(force=True)
                except Exception:
                    pass


# ===========================================================================
# Asynchronous Engine (`AsyncRevolver` — Phase 2)
# ===========================================================================

class AsyncRevolver(_BaseRevolver):
    """
    Native Asynchronous Revolver Engine powered by `asyncio` and `httpx.AsyncClient`.

    Eliminates thread-pool overhead for high-concurrency async applications (FastAPI,
    Discord/Telegram bots, async workers) and cancels losing HTTP requests mid-flight.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful, concise, and intelligent AI assistant.",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        judge_max_attempts: int = 2,
        judge_instruction: Optional[str] = None,
        corruption_markers: Optional[Tuple[str, ...]] = None,
        max_keepalive_connections: int = DEFAULT_ASYNC_MAX_KEEPALIVE,
        max_connections: int = DEFAULT_ASYNC_MAX_CONNECTIONS,
        race_mode: str = DEFAULT_RACE_MODE,
        hedge_delay: Optional[float] = None,
        min_hedge_delay: float = MIN_HEDGE_DELAY,
        max_hedge_delay: float = MAX_HEDGE_DELAY,
        hedge_multiplier: float = HEDGE_LATENCY_MULTIPLIER,
        cost_rates: Optional[Dict[str, Tuple[float, float]]] = None,
        budget_guard: Optional["BudgetGuard"] = None,
        state_path: Optional[str] = None,
        persist_state: bool = True,
    ):
        if httpx is None:
            raise ImportError("AsyncRevolver requires 'httpx'. Install it with: pip install httpx")

        super().__init__(
            system_prompt=system_prompt,
            timeout=timeout,
            judge_max_attempts=judge_max_attempts,
            judge_instruction=judge_instruction,
            corruption_markers=corruption_markers,
            race_mode=race_mode,
            hedge_delay=hedge_delay,
            min_hedge_delay=min_hedge_delay,
            max_hedge_delay=max_hedge_delay,
            hedge_multiplier=hedge_multiplier,
            cost_rates=cost_rates,
            budget_guard=budget_guard,
            state_path=state_path,
            persist_state=persist_state,
        )
        self._async_limits = httpx.Limits(
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
        )
        self._client: Optional["httpx.AsyncClient"] = None
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None

    async def __aenter__(self) -> "AsyncRevolver":
        self._get_client()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    def _get_client(self) -> "httpx.AsyncClient":
        """
        Return an `httpx.AsyncClient` bound to the active running event loop.

        Reuses explicitly injected clients (`_client_loop is None`) or recreates
        closed/stale clients if a new event loop starts (e.g., across multiple `asyncio.run` calls).
        """
        loop = asyncio.get_running_loop()
        if self._client is not None:
            is_closed = getattr(self._client, "is_closed", False)
            if not is_closed and (self._client_loop is None or self._client_loop is loop):
                return self._client

        self._client = httpx.AsyncClient(limits=self._async_limits)
        self._client_loop = loop
        return self._client

    def add_provider(
        self,
        provider: str,
        api_key: str,
        models: Optional[List[str]] = None,
        url: Optional[str] = None,
        models_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        tier: Optional[str] = None,
    ) -> List[str]:
        """Register a provider synchronously using `models` or known defaults."""
        p_name, target_url, m_url, headers, default_models = self._resolve_provider_config(
            provider, url, models_url, extra_headers
        )
        target_models = list(models) if models else default_models
        return self._register_provider_models(
            p_name, target_url, api_key, headers, target_models, provider, tier=tier,
            models_url=m_url,
        )

    async def adiscover_models(self, models_url: str, api_key: str) -> List[str]:
        """Asynchronously query provider's /models endpoint and return normalized model IDs."""
        headers = {"Authorization": f"Bearer {api_key}"}
        client = self._get_client()
        try:
            resp = await client.get(models_url, headers=headers, timeout=8.0)
            if resp.status_code == 200:
                return self._parse_discovered_models(resp.json(), max_chambers=50)
        except Exception as exc:
            log.warning("Async model discovery failed on %s: %s", models_url, exc)
        return []

    async def aadd_provider(
        self,
        provider: str,
        api_key: str,
        models: Optional[List[str]] = None,
        url: Optional[str] = None,
        models_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        discover_models: bool = False,
        max_chambers: int = 5,
        tier: Optional[str] = None,
    ) -> List[str]:
        """Asynchronously register a provider, optionally auto-discovering live models."""
        p_name, target_url, m_url, headers, default_models = self._resolve_provider_config(
            provider, url, models_url, extra_headers
        )

        target_models: List[str] = list(models) if models else []
        if not target_models and discover_models and m_url:
            client = self._get_client()
            headers_auth = {"Authorization": f"Bearer {api_key}"}
            try:
                resp = await client.get(m_url, headers=headers_auth, timeout=8.0)
                if resp.status_code == 200:
                    target_models = self._parse_discovered_models(resp.json(), max_chambers=max_chambers)
            except Exception as exc:
                log.warning("Async model discovery failed on %s: %s", m_url, exc)

        if not target_models:
            target_models = default_models

        return self._register_provider_models(
            p_name, target_url, api_key, headers, target_models, provider, tier=tier,
            models_url=m_url,
        )

    async def _acall_chamber(
        self,
        chamber: Dict[str, Any],
        messages: List[Dict[str, str]],
        timeout: Optional[float] = None,
    ) -> Tuple[Optional[str], Tuple[Any, ...]]:
        """Fire a single asynchronous chamber request over `httpx.AsyncClient`."""
        headers, payload, req_timeout = self._prepare_request(chamber, messages, timeout)
        client = self._get_client()
        started = time.monotonic()
        try:
            resp = await client.post(chamber["url"], headers=headers, json=payload, timeout=req_timeout)
        except httpx.TimeoutException:
            return None, ("timeout", None)
        except httpx.HTTPError as exc:
            log.warning("Async chamber %s network exception: %s", chamber["name"], exc)
            return None, ("network", None)

        elapsed = time.monotonic() - started
        return self._evaluate_response(chamber, resp, elapsed)

    @staticmethod
    async def _acancel_tasks(tasks: Set[asyncio.Task]) -> None:
        """Cancel pending tasks and await unwind cleanup so no tasks/exceptions dangle."""
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _aawait_tasks_winner(
        self,
        task_map: Dict[asyncio.Task, Dict[str, Any]],
        deadline: float,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Await first task returning valid content; cancel slow racers immediately mid-flight."""
        winner_content: Optional[str] = None
        winner_name: Optional[str] = None
        pending: Set[asyncio.Task] = set(task_map.keys())

        try:
            while pending:
                rem = deadline - time.monotonic()
                if rem <= 0.0:
                    log.warning("Async revolver race reached deadline across %d chamber(s)", len(task_map))
                    break

                wait_timeout = min(rem, self.timeout + 2.0)
                done, pending = await asyncio.wait(
                    pending, timeout=wait_timeout, return_when=asyncio.FIRST_COMPLETED
                )
                if not done:
                    log.warning("Async revolver race timed out across %d chamber(s)", len(task_map))
                    break

                # Drain every completed task in this batch to avoid unretrieved exceptions
                for task in done:
                    chamber = task_map[task]
                    try:
                        content, outcome = task.result()
                    except asyncio.CancelledError:
                        continue
                    except Exception as exc:
                        log.warning("Async racer %s raised: %s", chamber["name"], exc)
                        content, outcome = None, ("exception", None)
                    self._apply_outcome(chamber, outcome)
                    if content and winner_content is None:
                        winner_content = content
                        winner_name = chamber["name"]

                if winner_content is not None:
                    break
        finally:
            await self._acancel_tasks(pending)

        return winner_content, winner_name

    async def _arace_concurrent(
        self,
        chambers: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        deadline: float,
    ) -> Tuple[Optional[str], Optional[str], Set[str]]:
        """Immediate async concurrent race across candidate chambers."""
        attempted: Set[str] = set()
        if not chambers:
            return None, None, attempted

        if len(chambers) == 1:
            attempted.add(chambers[0]["name"])
            req_timeout = max(0.1, min(self.timeout, deadline - time.monotonic()))
            content, outcome = await self._acall_chamber(chambers[0], messages, timeout=req_timeout)
            self._apply_outcome(chambers[0], outcome)
            with self._health_lock:
                self._metrics["single_chamber_calls"] += 1
            return content, (chambers[0]["name"] if content else None), attempted

        with self._health_lock:
            self._metrics["concurrent_races"] += 1

        req_timeout = max(0.1, min(self.timeout, deadline - time.monotonic()))
        task_map: Dict[asyncio.Task, Dict[str, Any]] = {}
        for chamber in chambers:
            attempted.add(chamber["name"])
            t = asyncio.create_task(self._acall_chamber(chamber, messages, timeout=req_timeout))
            task_map[t] = chamber

        winner_content, winner_name = await self._aawait_tasks_winner(task_map, deadline)
        return winner_content, winner_name, attempted

    async def _arace_hedged(
        self,
        chambers: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        deadline: float,
        override_hedge_delay: Optional[float] = None,
    ) -> Tuple[Optional[str], Optional[str], Set[str]]:
        """Async Staggered Hedged Race: awaits primary up to `delta_t` before spawning backup."""
        attempted: Set[str] = set()
        if not chambers:
            return None, None, attempted
        if len(chambers) == 1:
            return await self._arace_concurrent(chambers, messages, deadline)

        c1, c2 = chambers[0], chambers[1]
        delta_t = self.get_hedge_delay(c1, override_delay=override_hedge_delay)
        if delta_t <= 0.0:
            return await self._arace_concurrent([c1, c2], messages, deadline)

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None, None, attempted

        attempted.add(c1["name"])
        req_timeout_1 = max(0.1, min(self.timeout, remaining))
        t1 = asyncio.create_task(self._acall_chamber(c1, messages, timeout=req_timeout_1))

        wait_budget = min(delta_t, max(0.05, remaining))
        try:
            done, _ = await asyncio.wait([t1], timeout=wait_budget)
        except asyncio.CancelledError:
            await self._acancel_tasks({t1})
            raise

        if t1 in done:
            try:
                content1, outcome1 = t1.result()
            except Exception as exc:
                log.warning("Async primary racer %s raised: %s", c1["name"], exc)
                content1, outcome1 = None, ("exception", None)
            self._apply_outcome(c1, outcome1)

            if content1:
                with self._health_lock:
                    self._metrics["hedged_saved_calls"] += 1
                log.info("Async hedged win (saved backup call): %s", c1["name"])
                return content1, c1["name"], attempted

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None, None, attempted

            attempted.add(c2["name"])
            with self._health_lock:
                self._metrics["hedge_triggers"] += 1
            log.info(
                "Async primary %s failed fast (%s); firing backup %s immediately",
                c1["name"], outcome1[0], c2["name"],
            )
            req_timeout_2 = max(0.1, min(self.timeout, remaining))
            t2 = asyncio.create_task(self._acall_chamber(c2, messages, timeout=req_timeout_2))
            winner_content, winner_name = await self._aawait_tasks_winner({t2: c2}, deadline)
            return winner_content, winner_name, attempted

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            await self._acancel_tasks({t1})
            return None, None, attempted

        attempted.add(c2["name"])
        with self._health_lock:
            self._metrics["hedge_triggers"] += 1
        log.info("Async hedge triggered after %.2fs: firing %s alongside %s", delta_t, c2["name"], c1["name"])

        req_timeout_2 = max(0.1, min(self.timeout, remaining))
        t2 = asyncio.create_task(self._acall_chamber(c2, messages, timeout=req_timeout_2))
        winner_content, winner_name = await self._aawait_tasks_winner({t1: c1, t2: c2}, deadline)
        return winner_content, winner_name, attempted

    async def _ajudge_repair(
        self,
        ranked_chambers: List[Dict[str, Any]],
        draft: str,
        reason: str,
        custom_judge_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Invoke the Judge asynchronously on up to `judge_max_attempts` chambers."""
        if not ranked_chambers:
            return None

        judge_messages = self._build_judge_messages(draft, reason, custom_judge_instruction)
        for judge in ranked_chambers[: self.judge_max_attempts]:
            log.info("Async Revolver Judge invoked on %s (reason: %s)", judge["name"], reason)
            content, outcome = await self._acall_chamber(judge, judge_messages)
            self._apply_outcome(judge, outcome)
            if content:
                cleaned = self.sanitize_draft(content)
                if cleaned:
                    log.info("Async Revolver Judge successfully repaired draft via %s", judge["name"])
                    return cleaned
        return None

    async def _afinalize_draft(
        self,
        ranked_chambers: List[Dict[str, Any]],
        draft: str,
        custom_judge_instruction: Optional[str] = None,
        response_format: Any = None,
    ) -> Optional[str]:
        """Validate winning draft; mechanically sanitize or escalate asynchronously to the Judge.

        When ``response_format`` is a Pydantic model class, the draft is also validated
        against the JSON schema and auto-repaired by The Schema Judge when invalid.
        """
        clean_or_none, fallback_draft, reason = self._try_sanitize_without_judge(draft)
        if clean_or_none is not None:
            candidate = clean_or_none
        else:
            repaired = await self._ajudge_repair(ranked_chambers, fallback_draft, reason, custom_judge_instruction)
            candidate = self._accept_repaired_draft(repaired)
            if candidate is None:
                return None

        if response_format is not None:
            return await self._aapply_schema_validation(ranked_chambers, candidate, response_format, custom_judge_instruction)
        return candidate

    async def _aapply_schema_validation(
        self,
        ranked_chambers: List[Dict[str, Any]],
        draft: str,
        response_format: Any,
        custom_judge_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Async variant of `_apply_schema_validation`: repair draft to conform to Pydantic schema."""
        instruction_template = custom_judge_instruction or DEFAULT_JSON_JUDGE_INSTRUCTION
        current = draft
        for _ in range(max(1, self.judge_max_attempts)):
            obj, error = self._validate_schema(current, response_format)
            if obj is not None:
                with self._health_lock:
                    self._metrics["schema_repairs"] += 1
                return self._extract_json_text(current)  # already valid JSON, strip fences

            for judge in ranked_chambers[: self.judge_max_attempts]:
                log.info("Async Revolver Schema Judge invoked on %s (error: %s)", judge["name"], error)
                repair_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "system", "content": instruction_template.format(error=error)},
                    {"role": "user", "content": f"Invalid draft to repair:\n{current}"},
                ]
                content, outcome = await self._acall_chamber(judge, repair_messages)
                self._apply_outcome(judge, outcome)
                if content:
                    cleaned = self.sanitize_draft(content)
                    candidate = self._extract_json_text(cleaned or content)
                    obj2, _ = self._validate_schema(candidate, response_format)
                    if obj2 is not None:
                        with self._health_lock:
                            self._metrics["schema_repairs"] += 1
                        return candidate
                    current = candidate

        with self._health_lock:
            self._metrics["schema_failures"] += 1
        return None

    async def ask(
        self,
        user_message: str,
        extra_context: str = "",
        system_prompt: Optional[str] = None,
        judge_instruction: Optional[str] = None,
        race_mode: Optional[str] = None,
        hedge_delay: Optional[float] = None,
        stream: bool = False,
        response_format: Any = None,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Process prompt asynchronously through Staggered Hedging / Concurrent Race + Judge.

        When ``stream=True``, returns an async generator yielding content chunks live (TTFT race)
        instead of a single completed string. The Judge repair loop is skipped for streaming.
        When ``response_format`` is a Pydantic model class, the reply is validated against
        its JSON schema and auto-healed by The Schema Judge.
        """
        if stream:
            return self._aask_streaming(
                user_message,
                extra_context=extra_context,
                system_prompt=system_prompt,
                judge_instruction=judge_instruction,
                race_mode=race_mode,
                hedge_delay=hedge_delay,
            )
        if not self._chambers:
            return NO_CHAMBERS_MESSAGE

        active_mode = self._validate_race_mode(race_mode or self.race_mode)
        with self._health_lock:
            self._metrics["total_requests"] += 1

        messages = self._build_messages(user_message, extra_context, system_prompt)
        deadline = time.monotonic() + self.timeout * 2
        healthy, cooled = self.rank_chambers(time.time())
        ranked = healthy + cooled

        classification = self.classify_prompt(user_message)
        attempted_names: Set[str] = set()
        racers = self._select_racers(healthy, cooled, classification)

        # Budget guard: if paid racers are blocked, fall back to free-only chambers.
        budget_exhausted = False
        if racers and self._budget_guard is not None:
            paid_racers = [c for c in racers if self._budget_guard.is_paid(c["name"])]
            if paid_racers:
                try:
                    self._budget_check()
                except BudgetExceeded:
                    log.warning("Async budget exhausted; skipping paid chambers, falling back to free-only")
                    budget_exhausted = True
                    racers = [c for c in racers if not self._budget_guard.is_paid(c["name"])]
                    if not racers:
                        return BUSY_MESSAGE + " (budget exhausted, no free chambers available)"

        if racers:
            if active_mode == "concurrent":
                content, _, attempted_names = await self._arace_concurrent(racers, messages, deadline)
            else:
                content, _, attempted_names = await self._arace_hedged(
                    racers, messages, deadline, override_hedge_delay=hedge_delay
                )
            if content:
                final = await self._afinalize_draft(ranked, content, judge_instruction, response_format=response_format)
                if final:
                    return final

        for chamber in healthy + cooled:
            if chamber["name"] in attempted_names:
                continue
            # Budget guard: skip paid chambers if budget is exhausted
            if self._budget_guard is not None and self._budget_guard.is_paid(chamber["name"]):
                if budget_exhausted:
                    continue
                try:
                    self._budget_check()
                except BudgetExceeded:
                    log.warning(
                        "Async budget exhausted; skipping paid chamber %s in fallback loop", chamber["name"]
                    )
                    budget_exhausted = True
                    continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            req_timeout = max(0.1, min(self.timeout, remaining))
            content, outcome = await self._acall_chamber(chamber, messages, timeout=req_timeout)
            self._apply_outcome(chamber, outcome)
            attempted_names.add(chamber["name"])
            if content:
                final = await self._afinalize_draft(ranked, content, judge_instruction, response_format=response_format)
                if final:
                    return final

        return BUSY_MESSAGE

    async def _aask_streaming(
        self,
        user_message: str,
        extra_context: str = "",
        system_prompt: Optional[str] = None,
        judge_instruction: Optional[str] = None,
        race_mode: Optional[str] = None,
        hedge_delay: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Async stream a reply via TTFT race: first chamber to yield a chunk wins; losers cancelled."""
        if not self._chambers:
            yield NO_CHAMBERS_MESSAGE
            return

        active_mode = self._validate_race_mode(race_mode or self.race_mode)
        del active_mode  # streaming always fires both racers concurrently for a true TTFT race
        with self._health_lock:
            self._metrics["total_requests"] += 1

        messages = self._build_messages(user_message, extra_context, system_prompt)
        deadline = time.monotonic() + self.timeout * 2
        healthy, cooled = self.rank_chambers(time.time())
        classification = self.classify_prompt(user_message)
        racers = self._select_racers(healthy, cooled, classification)
        racers_map = {c["name"]: c for c in racers}

        result_q: "asyncio.Queue[Tuple[str, ...]]" = asyncio.Queue()
        stop_events: Dict[str, asyncio.Event] = {}
        tasks: Dict[str, asyncio.Task] = {}

        async def _run(chamber: Dict[str, Any]) -> None:
            name = chamber["name"]
            stop = stop_events[name]
            headers, payload, req_timeout = self._prepare_request(chamber, messages)
            headers["Accept"] = "text/event-stream"
            payload["stream"] = True
            client = self._get_client()
            resp = None
            try:
                async with client.stream(
                    "POST", chamber["url"], headers=headers, json=payload, timeout=req_timeout
                ) as resp:
                    async for raw_line in resp.aiter_lines():
                        if stop.is_set():
                            break
                        if not raw_line:
                            continue
                        chunk = self._parse_sse_delta(raw_line)
                        usage = self._parse_sse_usage(raw_line)
                        if chunk:
                            await result_q.put(("chunk", name, chunk))
                        if usage:
                            await result_q.put(("usage", name, usage))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("Async streaming chamber %s errored: %s", name, exc)
                await result_q.put(("error", name, str(exc)))
            finally:
                await result_q.put(("done", name))

        for chamber in racers:
            stop_events[chamber["name"]] = asyncio.Event()
            tasks[chamber["name"]] = asyncio.create_task(_run(chamber))

        winner: Optional[str] = None
        finished: Set[str] = set()
        yielded_any = False
        started = time.monotonic()
        winner_usage: Optional[Dict[str, int]] = None
        try:
            while len(finished) < len(racers):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    log.warning("Async streaming race reached deadline across %d chamber(s)", len(racers))
                    break
                try:
                    item = await asyncio.wait_for(result_q.get(), timeout=min(0.2, remaining))
                except asyncio.TimeoutError:
                    continue
                kind = item[0]
                if kind == "done":
                    finished.add(item[1])
                    continue
                if kind == "error":
                    finished.add(item[1])
                    continue
                if kind == "usage":
                    _, name, usage = item
                    if winner is not None and name == winner:
                        winner_usage = usage
                    continue
                if kind == "chunk":
                    _, name, chunk = item
                    if winner is None:
                        winner = name
                        yielded_any = True
                        with self._health_lock:
                            self._metrics["concurrent_races"] += 1
                        for other_name, ev in stop_events.items():
                            if other_name != name:
                                ev.set()
                    if winner == name:
                        yield chunk
                        yielded_any = True
        finally:
            for ev in stop_events.values():
                ev.set()
            await self._acancel_tasks(set(tasks.values()))

        if winner is not None and yielded_any:
            elapsed = time.monotonic() - started
            self._apply_outcome(racers_map[winner], ("ok", elapsed, winner_usage))
        elif racers:
            for chamber in racers:
                if chamber["name"] != winner:
                    self._apply_outcome(chamber, ("network", None))

    async def aclose(self) -> None:
        """Asynchronously close the pooled `httpx.AsyncClient`."""
        if self._client is not None:
            try:
                if not getattr(self._client, "is_closed", False):
                    await self._client.aclose()
            except Exception as exc:
                log.warning("Error during AsyncRevolver aclose: %s", exc)
            finally:
                self._client = None
                self._client_loop = None
        if self._state_store is not None:
            try:
                self._state_store.save(force=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Module-level Default Instance & Convenience Shortcuts
# ---------------------------------------------------------------------------
default_revolver = Revolver()


def add_chamber(*args, **kwargs):
    return default_revolver.add_chamber(*args, **kwargs)


def add_provider(*args, **kwargs):
    return default_revolver.add_provider(*args, **kwargs)


def ask(user_message: str, *args, **kwargs) -> Union[str, Generator[str, None, None]]:
    return default_revolver.ask(user_message, *args, **kwargs)


def get_status() -> List[Dict[str, Any]]:
    return default_revolver.get_status()


def get_metrics() -> Dict[str, Any]:
    return default_revolver.get_metrics()


def export_prometheus() -> str:
    return default_revolver.export_prometheus()


def set_budget(limit_usd: float, cycle_type: str = "monthly", custom_days: int = 30, warning_threshold_pct: float = 80.0, config_path: Optional[str] = None) -> None:
    return default_revolver.set_budget(limit_usd, cycle_type, custom_days, warning_threshold_pct, config_path)


def mark_chamber_paid(name: str, rate_prompt: float, rate_completion: float) -> None:
    return default_revolver.mark_chamber_paid(name, rate_prompt, rate_completion)


def mark_chamber_free(name: str) -> None:
    return default_revolver.mark_chamber_free(name)


def get_budget_status() -> Dict[str, Any]:
    return default_revolver.get_budget_status()


# --- Universal provider auto-detection & dynamic discovery (Phase 6) -------

def refresh_provider(*args, **kwargs):
    return default_revolver.refresh_provider(*args, **kwargs)


def prune_dead_models(*args, **kwargs):
    return default_revolver.prune_dead_models(*args, **kwargs)


def replace_pruned_with_healthy(*args, **kwargs):
    return default_revolver.replace_pruned_with_healthy(*args, **kwargs)


def get_chamber_outage(name: str) -> Dict[str, Any]:
    return default_revolver.get_chamber_outage(name)
