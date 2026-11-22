#!/usr/bin/env python3
"""Turn a midsem matrix .bin (gen_matrices.py) into data.S for bare-metal Ara.
usage: bin2S.py m.bin [--no-dense] [--window W] [--spmm K] > data.S
--no-dense emits a 1-element DENSE (for builds that skip the dense kernels, so n=256 still fits the 1 MB testbench RAM).
--window W sets the segmented-reduction window: W = VLMAX at e32/LMUL 1 = VLEN/32 (default 128, VLEN 4096).
--spmm K (second workload, spmm.c): C = A x B with a dense n x K matrix B.
Emits K, SPMM_B (B column by column, each column laid out like x), SPMM_CREF (reference C = A @ B in float64,
row-major n x K) and SPMM_C (output). B is drawn with a fixed seed, so runs are reproducible."""
import struct, sys
import numpy as np
b = open(sys.argv[1], 'rb').read()
R, C, NZ = struct.unpack_from('<3i', b, 0); o = 12
def take(n, t):
    global o
    a = np.frombuffer(b, dtype=t, count=n, offset=o); o += n * 4; return a
rp, ci = take(R + 1, '<i4'), take(NZ, '<i4')
val, x, yref = take(NZ, '<f4'), take(C, '<f4'), take(R, '<f4')
dense = np.zeros((R, C), dtype='<f4')  # filled below unless --no-dense
no_dense = '--no-dense' in sys.argv
for r in range(R):
    dense[r, ci[rp[r]:rp[r + 1]]] = val[rp[r]:rp[r + 1]]
W = int(sys.argv[sys.argv.index('--window') + 1]) if '--window' in sys.argv else 128
LOGW = W.bit_length() - 1
assert W == 1 << LOGW and W % 8 == 0, 'window must be a power of two, at least 8'

def segred_tables(rp, nnz, W):
    """One-time preprocessing for the segmented-reduction kernels (segred_hw in main.c, segred_hw_mm in spmm.c).
    The flat nonzero list is cut into windows of W elements, ignoring row boundaries.
    nnz = nonzeros rounded up to a multiple of 8: the extra elements are zero padding with no tail
    flag, because indexed accesses with an odd vl made this Ara raise misaligned-address traps."""
    lens = np.diff(rp)
    nonempty = np.nonzero(lens)[0]
    tail = np.zeros(nnz, dtype=np.uint8)              # 1 = last nonzero of its row (padding: 0)
    tail[rp[nonempty + 1] - 1] = 1
    seg = np.concatenate([[0], np.cumsum(tail)[:-1]]) if nnz else tail   # row-piece id of each element
    nwin = -(-nnz // W)
    pad = nwin * W - nnz
    tail_p, seg_p = np.concatenate([tail, np.zeros(pad, np.uint8)]), np.concatenate([seg, np.full(pad, -1)])
    info = np.zeros((nwin, 3), dtype=np.int32)         # per window: results k, log2 of longest row piece, carry out
    for w in range(nwin):
        s = seg_p[w * W:(w + 1) * W]
        n = min(W, nnz - w * W)
        piece = int(np.max(np.unique(s[:n], return_counts=True)[1]))  # longest row piece in this window
        info[w] = (tail_p[w * W:w * W + n].sum(), (piece - 1).bit_length(), 0 if tail_p[w * W + n - 1] else 1)
    bits = lambda a: np.packbits(a.reshape(-1, 8), axis=1, bitorder='little').ravel()  # RVV mask: bit i = element i
    # Byte offset in y of each element's row (32-bit). Only the offsets at row ends are used:
    # the kernel scatters with the tail flags as mask (no packing needed).
    # 16-bit offsets would halve the memory but made Ara fault.
    doff = np.zeros(nnz, dtype=np.uint32); doff[:rp[-1]] = 4 * np.repeat(np.arange(len(lens)), lens)
    return bits(tail_p) if nwin else np.zeros(0, np.uint8), info.ravel(), doff, nwin

spmm_k = int(sys.argv[sys.argv.index('--spmm') + 1]) if '--spmm' in sys.argv else 0
if spmm_k:
    rng = np.random.default_rng(2026)
    spmm_b = rng.uniform(-1, 1, size=(spmm_k, C)).astype('<f4')            # [column j][row r]
    a64 = np.zeros((R, C)); [a64.__setitem__((r, ci[rp[r]:rp[r + 1]]), val[rp[r]:rp[r + 1]]) for r in range(R)]
    spmm_cref = (a64 @ spmm_b.T.astype(np.float64)).astype('<f4')          # n x K, row-major

def emit(name, a):
    print(f'.global {name}\n.balign 64\n{name}:')
    raw = a.tobytes(); raw += bytes(-len(raw) % 4)   # pad byte arrays to whole words
    w = np.frombuffer(raw, dtype='<u4')
    for i in range(0, len(w), 8):
        print('  .word ' + ','.join(hex(v) for v in w[i:i + 8]))
print('.section .l2,"aw",@progbits')
for n, v in (('R', R), ('C', C), ('NZ', NZ)):
    print(f'.global {n}\n.balign 8\n{n}: .word {v}')
NZP = -(-NZ // 8) * 8   # segred_hw reads VAL/COL_OFF up to NZP (multiple of 8): pad with zeros
padz = lambda a: np.concatenate([a, np.zeros(NZP - NZ, a.dtype)])
for n, a in (('ROW_PTR', rp), ('COL_IDX', ci), ('COL_OFF', padz(ci * 4)), ('VAL', padz(val)), ('X', x), ('Y_REF', yref), ('DENSE', dense.ravel()[:1] if no_dense else dense.ravel()), ('Y', np.zeros(R, dtype='<f4'))):
    emit(n, a)
seg_tail, seg_info, seg_doff, seg_nwin = segred_tables(rp, NZP, W)
for n, v in (('SEG_NNZ', NZP), ('SEG_W', W), ('SEG_NWIN', seg_nwin)):
    print(f'.global {n}\n.balign 8\n{n}: .word {v}')
for n, a in (('SEG_TAIL', seg_tail), ('SEG_INFO', seg_info), ('SEG_DOFF', seg_doff)):
    emit(n, a)
if spmm_k:
    print(f'.global K\n.balign 8\nK: .word {spmm_k}')
    for n, a in (('SPMM_B', spmm_b.ravel()), ('SPMM_CREF', spmm_cref.ravel()), ('SPMM_C', np.zeros(R * spmm_k, dtype='<f4'))):
        emit(n, a)
