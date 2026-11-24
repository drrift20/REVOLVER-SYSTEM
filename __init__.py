"""
Revolver System — Self-Healing, Staggered-Hedged, Sync & Async Judge Revolver Engine.
"""

from .revolver import (
    Revolver,
    AsyncRevolver,
    add_chamber,
    add_provider,
    ask,
    get_status,
    get_metrics,
    default_revolver,
    DEFAULT_JUDGE_INSTRUCTION,
    DEFAULT_CORRUPTION_MARKERS,
    DEFAULT_RACE_MODE,
    DEFAULT_HEDGE_DELAY,
    MIN_HEDGE_DELAY,
    MAX_HEDGE_DELAY,
    DEFAULT_ASYNC_MAX_KEEPALIVE,
    DEFAULT_ASYNC_MAX_CONNECTIONS,
    KNOWN_PROVIDERS,
)

__version__ = "1.2.0"
__all__ = [
    "Revolver",
    "AsyncRevolver",
    "add_chamber",
    "add_provider",
    "ask",
    "get_status",
    "get_metrics",
    "default_revolver",
    "DEFAULT_JUDGE_INSTRUCTION",
    "DEFAULT_CORRUPTION_MARKERS",
    "DEFAULT_RACE_MODE",
    "DEFAULT_HEDGE_DELAY",
    "MIN_HEDGE_DELAY",
    "MAX_HEDGE_DELAY",
    "DEFAULT_ASYNC_MAX_KEEPALIVE",
    "DEFAULT_ASYNC_MAX_CONNECTIONS",
    "KNOWN_PROVIDERS",
]
