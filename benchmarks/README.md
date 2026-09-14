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

## qaoa/ -- EXPERIMENTAL, not yet a validated result
Early results looked promising (`maxcut_mobius.py` beating both Adam and
the raw-Newton default in one run), but re-running the identical
benchmark gave a losing result, and `max_independent_set.py`/
`max_independent_set_mobius.py` never won convincingly with either mode.
QAOA's curvature is confirmed genuinely sign-indefinite even in the
noiseless exact model (`d2_stability_diagnostic.py`) -- that diagnosis
is solid -- but the Mobius correction is not yet a reliable fix. These
scripts are included so you can see both the winning and losing runs
for yourself, not to claim a validated result. Do not use this domain
as a marketing example until it's resolved.

## bundle_adjustment/ -- camera calibration at scale
benchmark.py --config small_test|medium|realistic -- scales up to ~900
parameters (50 cameras, 200 points), where Levenberg-Marquardt's own
per-iteration cost starts to matter.

## rl_honest_null/ -- where SoftOpt does *not* help, shown honestly
Full PPO on LunarLander, both with the default newton mode and with
mobius. Both lose to plain Adam, as expected: full reinforcement
learning has a continuously moving policy target, not the fixed,
known objective SoftOpt's correction requires. Included so you can see
the honest limit for yourself, not just take our word for it.
