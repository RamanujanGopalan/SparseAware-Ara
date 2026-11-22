# Sparse-Aware Ara: segmented reduction for CSR SpMV (idea outline)

This is the idea behind the changes in this fork: where we started, what is slow, and what we propose to change in Ara.

## 1. Starting point

- **Workload:** sparse matrix × vector, y = A·x, with A stored in CSR (`val`, `col`, `row_ptr`). SpMM (A × dense B) follows the same pattern.
- **Platform:** Ara (4 lanes, RVV 1.0) attached to the CVA6 scalar core, simulated cycle-accurately in Verilator.
- **Baseline kernels on unmodified Ara:**
  - a dense RVV kernel: multiplies every element, zeros included;
  - a vector CSR kernel that works **one row at a time**: load the row's non-zeros, gather `x[col]`, multiply, then **reduce** (`vfredusum`) into `y[row]`.

## 2. The problem

- A vector reduction on Ara has a large **fixed cost per row**: about 30 cycles even for one element, plus the time for CVA6 to read the result back.
- At high sparsity, rows hold only a few non-zeros. So:
  - most lanes are idle during each short row;
  - most of the time goes into per-row overhead, not useful multiplies;
  - the dense kernel wastes nearly all its work on zeros.
- Ara's existing reduction can't simply be reused across rows. Each lane first adds up its own elements, and the elements of one row sit in different lanes, so rows packed together would be mixed.

```
row A = e0 e1 e2,  row B = e3 e4      lane 0 holds e0 and e4  ->  its partial sum mixes A and B
```

## 3. The idea: windows of non-zeros instead of rows

Treat CSR's flat non-zero list as one stream and process it in **fixed windows** (one full vector register), ignoring row boundaries.

1. **Element-wise work at full vector length:** load values, gather `x[col]`, multiply. This step doesn't need to know about rows.
2. **Tail flags (sparse metadata):** one bit per non-zero, set to 1 if it is the last non-zero of its row. Built once from `row_ptr` and loaded as the mask register `v0`.
3. **Segmented reduction:** add the products in order, restarting after every tail flag. This gives one total per row.
4. **Carry across windows:** a row cut by the window end passes its partial sum to the next window. This also covers rows longer than a window.
5. **Empty rows:** they have no non-zeros and no flag, produce nothing, and stay 0 in y.

```
A = [5 0 0 2]      non-zeros (flat):  5   2 | 1   3   4 | 6   7      x = [1, 2, 3, 4]
    [0 0 0 0]      products:          5   8 | 1   6  16 | 18  28
    [1 3 0 4]      tail flags:        0   1 | 0   0   1 | 0   1
    [0 0 6 7]      row totals:           13 |        23 |     46   -> y = [13, 0, 23, 46]
```

The cost per row (the reduction plus its scalar round trip) disappears. One window handles up to VLMAX non-zeros, however many rows it holds.

## 4. What we propose

### 4.1 First proposal: `vfsegredsum` (packed results)

| | |
|---|---|
| Inputs | products, tail flags, carry-in |
| Outputs | the finished row sums **packed** in row order, then the carry-out |
| Rule | keep adding; at each flag, output the sum and restart from 0 |

```c
acc = carry_in; j = 0;
for (i = 0; i < vl; i++) {
  acc += products[i];
  if (tail[i]) { out[j++] = acc; acc = 0; }
}
carry_out = acc;
```

A list of non-empty rows would map the j-th packed result to its row of y.

### 4.2 Refined proposal: `vfsegsum` (running sum at every position)

Checking the first proposal against Ara and a software prototype showed some problems:
- The carry-out lands at a position that changes from window to window, which makes the next window awkward to start.
- Packing needs `vcompress`-style logic, and Ara's `vcompress` had bugs on this path. We found and isolated 3 functional bugs while building the software version.
- The number of results per window has to be known in advance, or the scalar core waits on every window.

So the proposal became a simpler instruction that **doesn't pack**:

```
vfsegsum.vv vd, vs2, v0.t        (FP32; vsegsum.vv is the int32 version for bring-up)

acc = 0
for i in 0 .. vl-1:
    acc   = acc + vs2[i]
    vd[i] = acc
    if v0[i]: acc = 0
```

- Every position gets its running sum, so each row total sits at its tail position.
- A **masked scatter** (mask = `v0`, offsets = each element's row in y, precomputed) writes only the totals to y. This also places results correctly around empty rows without any packing.
- The carry is the last element of `vd`: one `vslidedown` moves it to element 0, and one `vfadd` adds it into the next window.
- It is a drop-in replacement for the slow step of the software version (log₂ rounds of slide + masked add).

### 4.3 Where it goes in the hardware

- **The mask unit (MASKU)** is the one block in Ara that already sees all lanes' elements back in order (it runs `viota`, a running count). The new instruction reuses that lane → MASKU → lane path, so no new wiring between blocks is needed.
- **New logic:**
  - FP32 adders (Ara's FP adders otherwise exist only inside the lanes);
  - a running-sum register that carries a row from one 8-element word to the next.
- **Small edits** in the decoder, op list, lane sequencer and lane ALU.
- **Timing concern:**
  - An FP add takes 4 cycles, so a single running-sum adder manages only 1 element every 4 cycles. That is slower than the gather feeding it.
  - Plan: a pipelined, flag-aware adder tree over the 8 elements the lanes deliver per cycle. It pays the 4-cycle wait once per 8 elements, not once per element.

## 5. How we plan to evaluate it

- **Correctness:** compare against scalar CSR on cases that stress the logic:
  - rows crossing windows;
  - rows longer than a window;
  - empty rows;
  - windows full of 1-element rows.
- **No regressions:** existing Ara programs must give identical results and identical cycle counts.
- **Performance:** cycles of SpMV and SpMM on synthetic matrices across sizes, structures (uniform, banded, triangular, half-empty, mixed) and sparsity from 0% to 99.9%. Compare against scalar CSR, dense RVV and vector CSR on unmodified Ara.
- **Expected outcome:**
  - wins at high sparsity, where rows are short and per-row overhead dominates;
  - losses at low sparsity, where every element pays for a gather and a scatter.
