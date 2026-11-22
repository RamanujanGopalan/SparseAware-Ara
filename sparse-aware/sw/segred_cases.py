#!/usr/bin/env python3
"""Edge-case matrices for the segmented-reduction kernel (segred_hw), in the same .bin format as
scripts/gen_matrices.py: int32 R, C, NZ | int32 row_ptr[R+1] | int32 col_idx[NZ] | float32 val[NZ]
| float32 x[C] | float32 y_ref[R]  (y_ref computed in float64, like the SciPy reference).
usage: segred_cases.py <outdir>      (window W = 128, the default for VLEN 4096)"""
import struct, sys
from pathlib import Path
import numpy as np

W = 128
rng = np.random.default_rng(7)

def write(path, lens, cols):
    lens = np.asarray(lens, dtype=np.int64)
    rp = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    ci = np.concatenate([np.sort(rng.choice(cols, l, replace=False)) for l in lens] or [[]]).astype(np.int32)
    val = rng.standard_normal(int(rp[-1])).astype(np.float32)
    x = rng.standard_normal(cols).astype(np.float32)
    y = np.array([np.dot(val[rp[r]:rp[r + 1]].astype(np.float64), x[ci[rp[r]:rp[r + 1]]].astype(np.float64))
                  for r in range(len(lens))], dtype=np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<3i', len(lens), cols, int(rp[-1])))
        for a in (rp, ci, val, x, y):
            f.write(a.tobytes())
    print(f'{path.name:28s} rows={len(lens):4d} cols={cols:4d} nnz={int(rp[-1]):6d}')

CASES = {
    # name: (row lengths, number of columns)
    'cross_windows':  (rng.integers(5, 13, 64), 64),         # many rows cut by window ends (carry)
    'longer_than_w':  (np.full(8, 300), 320),                # each row spans 3 windows (carry twice)
    'ends_at_edge':   (np.full(8, W), W),                    # every row ends exactly at a window end
    'empty_rows':     (rng.choice([0, 0, 0, 1, 2, 3], 96), 96),
    'all_1nnz_rows':  (np.ones(256, dtype=int), 256),        # 2 windows, 128 results each
    'all_empty':      (np.zeros(16, dtype=int), 16),         # no nonzeros at all
    'mixed':          (rng.choice([0, 1, 3, W - 1, W, W + 1, 3 * W + 5], 24), 400),
}

if __name__ == '__main__':
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'mats_segred')
    out.mkdir(parents=True, exist_ok=True)
    for name, (lens, cols) in CASES.items():
        write(out / f'segred_{name}.bin', lens, cols)
