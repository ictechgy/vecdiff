"""vecdiff — diff two embedding-index snapshots.

vecdiff compares two snapshots of an embedding index (blue/green of a
vector-DB migration, or now-vs-later of a rot audit) and reports graded
findings: neighbor stability (N1), population stats (N2), duplicates (N4).

Fully local, zero network at runtime, deterministic, numpy-only.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
