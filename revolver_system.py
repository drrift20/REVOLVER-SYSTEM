"""
revolver_system — Package compatibility module.

Re-exports the full public API of Revolver System so that both
`import revolver` and `import revolver_system` work identically.
"""

from revolver import *  # noqa: F401, F403
from revolver import __all__, __version__  # noqa: F401
