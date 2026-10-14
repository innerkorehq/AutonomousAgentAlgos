"""AutonomousAgentAlgos (AAA): event-driven agent algorithms over arbitrary data."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autonomous-agent-algos")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
