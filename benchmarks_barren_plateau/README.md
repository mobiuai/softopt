# Barren-plateau benchmark

An exact derivative helps in a barren plateau, and does not remove it. Both
claims are measured here.

```bash
pip install softopt
python3 plateau_scan.py          # 4-10 qubits, ~70 min
python3 plateau_replicate.py     # four falsification checks, ~25 min
```

`plateau_scan.py` sweeps the qubit count. As qubits are added the partial
derivatives shrink while shot noise does not, so the measurement degrades and
the exact derivative does not. `plateau_depth` in the output is shot noise
over the typical exact partial derivative: above 1 the measurement cannot
resolve the gradient at all.

`plateau_replicate.py` tries to break the result four ways -- fresh seed
streams, five shot budgets, five SPSA probe sizes, three different
Hamiltonians. Each one could overturn it. Measured result: SoftOpt covers
1.4-1.6x as much of the distance to the ground state as Adam, 316 wins in 320
runs across 16 conditions, every condition significant at p <= 4.8e-05.

Two things worth reading carefully.

The advantage **grows as the measurement degrades** -- 1.66 at 512 shots
against 1.44 at 65536. That direction was predicted before the run, and it is
the mechanism: the exact derivative matters most where the measured one is
least usable.

And at 10 qubits both arms covered only a few percent of the way to the ground
state. 1.5x of very little is still very little. This is a measurement of a
limit, not a solution to it.

The energy measurements are simulated shot noise on an exact statevector, not
a hardware noise model. The plateau is a property of the ansatz and does not
depend on the noise model; the FakeFez comparisons are elsewhere in
`benchmarks/`.
