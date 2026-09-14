# SoftOpt benchmarks

Every benchmark behind every number in the main README, organized by
domain. All use real hardware-noise-model data (IBM's FakeFez via Qiskit
+ Aer) or realistic finite-sample noise -- not idealized simulations --
and compare against genuine torch.optim.Adam.

```bash
pip install -e ..    # install softopt from the repo root
```

## vqe/ -- quantum chemistry
- h2.py, h4.py, c13cl2.py -- three molecules on a real EfficientSU2 ansatz
- efficient_su2_exact.py / efficient_su2_lemma61.py -- the shared, verified
  engine (bit-exact vs Qiskit's own Statevector) every molecule and spin-chain
  model below uses

## all_efficientsu2_models.py -- BeH2, HeH+, and six spin-chain models
Same verified engine as vqe/, different Hamiltonians:
```bash
python3 all_efficientsu2_models.py --model beh2
python3 all_efficientsu2_models.py --model heh
python3 all_efficientsu2_models.py --model ferro_ising
python3 all_efficientsu2_models.py --model transverse_ising
python3 all_efficientsu2_models.py --model xy_model
python3 all_efficientsu2_models.py --model af_heisenberg
python3 all_efficientsu2_models.py --model ssh
python3 all_efficientsu2_models.py --model kitaev
python3 all_efficientsu2_models.py --model all       # every model, one run
```

## spin_chains/ -- classical/quasi-periodic models
- quasicrystal.py -- Penrose P3 tiling, 50 sites, de Bruijn pentagrid construction
- fibonacci.py -- antiferromagnetic Fibonacci chain, N=16, exact ground state
  via 2000-restart L-BFGS-B

## grape/, camera_calibration/, portfolio/ -- the three flagship domains
The domains SoftOpt is being developed into full standalone products for.
GRAPE also includes a comparison against QuTiP's own specialized
GRAPE/L-BFGS-B optimizer as a reference point.

## bundle_adjustment/ -- camera calibration at scale
benchmark.py --config small_test|medium|realistic -- scales up to ~900
parameters (50 cameras, 200 points), where Levenberg-Marquardt's own
per-iteration cost starts to matter.
