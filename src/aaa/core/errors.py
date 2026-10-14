class AAAError(Exception):
    """Base class for all library errors."""


class ConcurrencyError(AAAError):
    """Optimistic-concurrency check failed (expected_seq mismatch)."""


class SchemaError(AAAError):
    """Unknown, invalid, or un-migratable payload schema."""


class StoreUnavailable(AAAError):
    """The event/blob store could not be reached within the retry budget (a transient outage that
    outlasted `RetryPolicy`). Never a reason to fail a run: the work is retried or redelivered."""


class Redeliver(AAAError):
    """Raised by a bus handler to say: "I could not finish this event because of a temporary problem;
    do NOT acknowledge it, deliver it again later". The engine raises it when a handler hit a store
    outage that outlasted its local retries. It is not an error in the run."""
