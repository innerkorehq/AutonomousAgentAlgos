"""Autonomous Agent Architecture (AAA): an event-driven architecture for production agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autonomous-agent-architecture")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
