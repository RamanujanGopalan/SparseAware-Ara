#!/usr/bin/env python3
# Matrix generator shared by the team project (unchanged copy).
"""Generate CSR test matrices and SciPy reference results for the SpMV kernels.

Writes data/<name>.bin in the format read by src/spmv.c:
  int32 rows, cols, nnz | int32 row_ptr[rows+1] | int32 col_idx[nnz]
  float32 val[nnz] | float32 x[cols] | float32 y_ref[rows]

y_ref is computed by scipy.sparse in float64 and stored as float32.

Structures (--structure):
  uniform    non-zeros anywhere, uniformly at random (data/uniform_n<n>_s<s>.bin)
  banded     non-zeros only within the narrowest band |i - j| <= w that has
             enough slots; every row gets about the same count, adjacent columns
  lower_tri  non-zeros only on/below the diagonal; row i has about (i+1)*p,
             so row lengths spread evenly from short to long
  half_rows  non-zeros only in the top n/2 rows; the bottom half is empty
  mixed      bimodal row lengths: a random 20% of rows ("heavy") are 16x as
             likely to hold a non-zero as the rest ("light"); at 90% sparsity
             that is ~40% density in heavy rows and ~2.5% in light rows.
             Made for the per-row dispatcher (ara_midsem DISPATCH_ELISION.md #2)
For the non-uniform structures the non-zeros are sampled uniformly inside the
allowed region, so every sparsity level is hit exactly (as long as the region
has enough slots: triangle and half-rows need sparsity >= ~50%).
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DEFAULT_SPARSITIES = [0, 50, 75, 90, 95, 99, 99.9]


def make_uniform(n, sparsity_pct, rng):
    """n x n matrix whose non-zeros are placed uniformly at random."""
    density = 1.0 - sparsity_pct / 100.0
    if density >= 1.0:
        A = sp.csr_matrix(rng.uniform(-1, 1, size=(n, n)))
    else:
        A = sp.random(n, n, density=density, format="csr", random_state=rng,
                      data_rvs=lambda k: rng.uniform(-1, 1, size=k))
    # Values drawn exactly at 0.0 would be dropped silently; nudge them.
    A.data[A.data == 0.0] = 0.5
    A.sort_indices()
    return A.astype(np.float32)


STRUCTURES = ["uniform", "banded", "lower_tri", "half_rows", "mixed"]
MIXED_HEAVY_FRAC, MIXED_RATIO = 0.2, 16


def make_mixed(n, sparsity_pct, rng):
    """n x n matrix with round(density * n^2) non-zeros; a random 20% of rows
    weigh MIXED_RATIO times more when the positions are drawn (no replacement,
    so a heavy row that fills up simply passes the rest to the light rows)."""
    nnz = int(round((1.0 - sparsity_pct / 100.0) * n * n))
    heavy = rng.choice(n, size=int(round(MIXED_HEAVY_FRAC * n)), replace=False)
    w = np.ones(n)
    w[heavy] = MIXED_RATIO
    p = np.repeat(w, n)
    pos = np.sort(rng.choice(n * n, size=nnz, replace=False, p=p / p.sum()))
    vals = rng.uniform(-1, 1, size=nnz)
    vals[vals == 0.0] = 0.5
    A = sp.csr_matrix((vals, (pos // n, pos % n)), shape=(n, n))
    A.sort_indices()
    return A.astype(np.float32)


def band_slots(n, w):
    """Entries with |i - j| <= w in an n x n matrix."""
    return n * (2 * w + 1) - w * (w + 1)


def allowed_region(n, structure, nnz):
    """Boolean n x n mask of positions a structure may use."""
    i, j = np.indices((n, n))
    if structure == "banded":
        w = 0
        while band_slots(n, w) < nnz:
            w += 1
        return np.abs(i - j) <= w
    if structure == "lower_tri":
        return j <= i
    if structure == "half_rows":
        return i < n // 2
    raise ValueError(structure)


def make_structured(n, sparsity_pct, structure, rng):
    """n x n matrix with round(density * n^2) non-zeros, sampled uniformly
    inside the structure's allowed region."""
    nnz = int(round((1.0 - sparsity_pct / 100.0) * n * n))
    slots = np.flatnonzero(allowed_region(n, structure, nnz))
    if nnz > len(slots):
        raise ValueError(f"{structure} cannot hold {nnz} non-zeros at n={n} "
                         f"(max {len(slots)}, i.e. sparsity >= "
                         f"{100 * (1 - len(slots) / n / n):.1f}%)")
    pos = np.sort(rng.choice(slots, size=nnz, replace=False))
    vals = rng.uniform(-1, 1, size=nnz)
    vals[vals == 0.0] = 0.5
    A = sp.csr_matrix((vals, (pos // n, pos % n)), shape=(n, n))
    A.sort_indices()
    return A.astype(np.float32)


def write_matrix(path, A, x):
    A = A.tocsr()
    y_ref = (A.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    with open(path, "wb") as f:
        np.array([A.shape[0], A.shape[1], A.nnz], dtype="<i4").tofile(f)
        A.indptr.astype("<i4").tofile(f)
        A.indices.astype("<i4").tofile(f)
        A.data.astype("<f4").tofile(f)
        x.astype("<f4").tofile(f)
        y_ref.astype("<f4").tofile(f)


def sparsity_tag(s):
    return f"{s:g}".replace(".", "p")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", type=int, default=512, help="matrix dimension (n x n)")
    p.add_argument("-s", "--sparsity", type=float, nargs="+",
                   default=DEFAULT_SPARSITIES, help="sparsity levels in percent")
    p.add_argument("--structure", choices=STRUCTURES, default="uniform")
    p.add_argument("--seed", type=int, default=511)
    p.add_argument("-o", "--outdir", default=Path(__file__).resolve().parent.parent / "data")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    x = rng.uniform(-1, 1, size=args.n).astype(np.float32)

    for s in args.sparsity:
        if args.structure == "uniform":
            A = make_uniform(args.n, s, rng)
        elif args.structure == "mixed":
            A = make_mixed(args.n, s, rng)
        else:
            A = make_structured(args.n, s, args.structure, rng)
        name = f"{args.structure}_n{args.n}_s{sparsity_tag(s)}"
        if A.nnz == 0:  # e.g. 8x8 at 99.9%: density * n^2 rounds to zero
            print(f"{name}: skipped (no non-zeros at this size)")
            continue
        write_matrix(outdir / f"{name}.bin", A, x)
        actual = 100.0 * (1 - A.nnz / (args.n * args.n))
        print(f"{name}: nnz={A.nnz} actual_sparsity={actual:.2f}%")


if __name__ == "__main__":
    main()
