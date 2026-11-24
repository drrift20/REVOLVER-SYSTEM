"""
Revolver System — Self-Healing, Multi-Trigger, Judge Revolver Engine.
"""

from .revolver import (
    Revolver,
    add_chamber,
    add_provider,
    ask,
    get_status,
    default_revolver,
    DEFAULT_JUDGE_INSTRUCTION,
    DEFAULT_CORRUPTION_MARKERS,
    KNOWN_PROVIDERS,
)

__version__ = "1.0.0"
__all__ = [
    "Revolver",
    "add_chamber",
    "add_provider",
    "ask",
    "get_status",
    "default_revolver",
    "DEFAULT_JUDGE_INSTRUCTION",
    "DEFAULT_CORRUPTION_MARKERS",
    "KNOWN_PROVIDERS",
]
