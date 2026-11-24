"""
Revolver System — Self-Healing, Staggered-Hedged, Judge Revolver Engine.
"""

from .revolver import (
    Revolver,
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
    KNOWN_PROVIDERS,
)

__version__ = "1.1.0"
__all__ = [
    "Revolver",
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
    "KNOWN_PROVIDERS",
]
