#!/usr/bin/env python3
"""Collect runs/*/sim.log into results/results.csv.

Each kernel prints one RES line; when the simulator has the testbench counters, the
"[roi-counters]" line printed when that kernel's measured region ends comes just before it.
"""
import csv, glob, re, sys

RUNS = sys.argv[1].rstrip('/') if len(sys.argv) > 1 else 'runs'   # e.g. runs_opt
OUT = 'results/results.csv' if RUNS == 'runs' else f'results/results_{RUNS}.csv'

FIELDS = ['cycles', 'ara_ar', 'ara_ar_bytes', 'ara_r_beats', 'ara_aw', 'ara_w_beats',
          'cva6_ar', 'cva6_r_beats', 'cva6_aw', 'cva6_w_beats', 'dc_loads', 'dc_misses']
rows = []
for f in sorted(glob.glob(f'{RUNS}/*/sim.log')):
    m = re.search(r'/(\w+?)_n(\d+)_s([0-9p]+)/', f)
    if not m:
        continue
    text = open(f, errors='ignore').read()
    ok = 'SUCCESS' in text
    counters = None
    for line in text.splitlines():
        if '[roi-counters]' in line:
            counters = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
        r = re.match(r'RES kernel=(\w+) n=(\d+) nnz=(\d+) cycles=(\d+) (\w+)', line)
        if r:
            row = dict(structure=m[1], n=int(r[2]), sparsity=float(m[3].replace('p', '.')), nnz=int(r[3]),
                       kernel=r[1], cycles=int(r[4]), passed=int(ok and r[5] == 'PASS'))
            if counters:
                row['roi_cycles'] = counters['cycles']
                row.update({k: counters[k] for k in FIELDS[1:]})
            rows.append(row)
            counters = None
rows.sort(key=lambda r: (r['structure'], r['n'], r['sparsity'], r['kernel']))
cols = ['structure', 'n', 'sparsity', 'nnz', 'kernel', 'cycles', 'passed', 'roi_cycles'] + FIELDS[1:]
with open(OUT, 'w', newline='') as o:
    w = csv.DictWriter(o, fieldnames=cols, restval='')
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} kernel runs, {sum(1 - r['passed'] for r in rows)} failed, "
      f"{sum('ara_ar' in r for r in rows)} with counters -> {OUT}")
