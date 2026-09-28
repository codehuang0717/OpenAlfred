"""Core package.

Import concrete modules directly (for example, ``core.database``). Keeping the
package initializer side-effect free prevents authentication/config import cycles.
"""

from .config import config

__all__ = ["config"]
