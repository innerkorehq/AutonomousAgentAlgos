class AAAError(Exception):
    """Base class for all library errors."""


class ConcurrencyError(AAAError):
    """Optimistic-concurrency check failed (expected_seq mismatch)."""


class SchemaError(AAAError):
    """Unknown, invalid, or un-migratable payload schema."""
