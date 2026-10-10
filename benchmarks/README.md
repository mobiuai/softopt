# SoftOpt benchmarks

Every benchmark behind every number in the main README, organized by
domain. All use real hardware-noise-model data (IBM's FakeFez via Qiskit
+ Aer) or realistic finite-sample noise -- not idealized simulations --
and compare against genuine torch.optim.Adam.

```bash
pip install -e ..    # install softopt from the repo root
```

## softbench/ -- one small problem per domain (calibrate=True, new in 0.7.0)
- softbench.py -- eleven domains (quantum chemistry, beam steering, robot arm, batch reactor, drug dosing,
  portfolio, building thermal control, battery charging, epidemic control, classifier under data shift,
  exact-model control), one protocol: a model with uncertain parameters, a real system that also carries an
  unmodelled effect, noisy measurements, 200 measurements, 10 seeds. Arms: Adam (best of five lrs), SPSA, COBYLA,
  SoftOpt default / measured / calibrate=True with and without bias parameters.
  `python3 softbench.py --problem pk` (or `--list`); `--bias_mult 3` reruns calibrate=True with a prior 3x too wide
- sb_savings.py -- measurements needed to reach Adam's 200-measurement quality (run after softbench.py)
- results/ -- the logs and per-seed results behind the README tables

## vqe/ -- quantum chemistry
- h2.py, h4.py, c13cl2.py -- three molecules on a real EfficientSU2 ansatz
- h4_real_molecule.py -- the H4 molecule itself (STO-3G, Jordan-Wigner, 8 qubits,
  185 Pauli terms; hamiltonians/h4_paulis.json), SoftOpt vs Adam, 10 seeds;
  soft_su2.py is the same Lemma 6.1 engine, vectorised for larger registers
- h2_vs_cobyla.py -- SoftOpt vs scipy's COBYLA on H2, same budget of device readings
- results/ -- the logs and per-seed results behind the README numbers
- efficient_su2_exact.py / efficient_su2_lemma61.py -- the shared, verified
  engine (bit-exact vs Qiskit's own Statevector) every molecule and spin-chain
  model below uses

## qml/ -- quantum machine learning
- vqc.py -- a variational quantum classifier trained on FakeFez: SoftOpt vs Adam vs
  COBYLA, same budget of device readings. --dataset iris (versicolor vs virginica,
  4 qubits, 24 parameters; run with --seeds 20) or --dataset cancer (breast cancer,
  8 PCA features, 8 qubits, 48 parameters). results/ holds the runs behind the README
  numbers (vqc_iris20_*, vqc_cancer_*)

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
