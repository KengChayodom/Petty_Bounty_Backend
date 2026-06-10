"""
Deterministic, hand-constructed embedding vectors for integration tests.

Cosine SIMILARITY = 1 - cosine_distance. For unit vectors a,b: distance is
1 - (a·b), so similarity == a·b. We use sparse axis-aligned vectors with known
dot products — NO randomness anywhere, so similarity outcomes are exactly
reproducible across runs.

    unit_vec(0)           ·  unit_vec(0)  = 1.0     -> similarity 1.0
    half_mix(0, 1)        ·  unit_vec(0)  = 1/√2    -> similarity ~0.70710678
    unit_vec(1)           ·  unit_vec(0)  = 0.0     -> similarity 0.0
"""
import math

DIM = 512

# Exact expected similarity of half_mix(i, j) against unit_vec(i).
HALF_MIX_SIMILARITY = 1.0 / math.sqrt(2)  # 0.7071067811865475


def unit_vec(i: int, dim: int = DIM) -> list[float]:
    """A unit vector with 1.0 at index i, zeros elsewhere."""
    v = [0.0] * dim
    v[i] = 1.0
    return v


def half_mix(i: int, j: int, dim: int = DIM) -> list[float]:
    """Unit vector splitting weight equally between axes i and j.
    Dot product with unit_vec(i) is exactly 1/√2."""
    v = [0.0] * dim
    c = 1.0 / math.sqrt(2)
    v[i] = c
    v[j] = c
    return v


def vec_literal(v: list[float]) -> str:
    """Render a Python vector as a pgvector text literal: '[a,b,c,...]'."""
    return "[" + ",".join(format(x, ".10g") for x in v) + "]"
