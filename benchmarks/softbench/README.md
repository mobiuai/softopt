# SoftBench

One small problem per domain, one protocol for all, for `calibrate=True`.

```bash
python3 softbench.py --list
python3 softbench.py --problem pk          # writes softbench_pk.json and prints the table
python3 sb_savings.py                      # measurements needed to reach Adam's 200-measurement quality
python3 softbench.py --problem pk --bias_mult 3   # calibrate=True with a prior 3x too wide
```

Each problem: `model(theta, beta)` in plain Python (soft-compiled), the real system = the model at the true
`beta` plus an effect the model does not have, noisy measurements, start at `theta = 0`, 200 measurements,
10 seeds. Score: fraction of the start gap to the true optimum remaining. `results/` holds the runs behind the
README tables.
