"""Exception types for vecdiff.

All are subclasses of VecdiffError; the CLI turns any of them into
exit code 3 (hard error), distinct from gate exit codes (0/1/2).
"""

from __future__ import annotations


class VecdiffError(Exception):
    """Base class for all vecdiff errors."""


class SnapshotError(VecdiffError):
    """A snapshot could not be loaded or failed metadata validation."""


class DimensionMismatchError(VecdiffError):
    """The two snapshots have different vector dimensions (hard error by spec)."""
