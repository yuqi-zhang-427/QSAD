# QSAD Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22105236.svg)](https://doi.org/10.5281/zenodo.22105236)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Reference implementation for *Quantum Sampling Architecture for Protein
Structure Reconstruction on Utility-Scale Hardware*
([arXiv:2607.06971](https://arxiv.org/abs/2607.06971)).

Quantum-sampled protein folding pipeline — from quantum bitstring
sampling to an all-atom PDB coordinate set. The prediction target is the
protein **backbone**; see [Scope of the reconstruction](#scope-of-the-reconstruction).

The prediction pipeline (Stages 1–3) is **code-only**: no sampling results
are bundled, and everything the pipeline consumes downstream is produced by
running Stage 1, which by default executes on the local `qiskit-aer` simulator
and needs no credentials of any kind.

Stage 4 is different. It reproduces the paper's energy-landscape figures
(Fig. 9 and Fig. 10) from **recorded utility-scale QPU runs** shipped in
`qpu_samples/` — 120-qubit and 49-qubit circuits executed on
`ibm_cleveland`. Those circuits are far beyond classical simulation, so the
recorded samples are the reproducible starting point; see
[Stage 4](#stage-4--energy-landscape-figures-fig-9-and-fig-10).

## Pipeline stages

```
tasks.csv
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1: Quantum sampling                                    │
│   entry: instance_sampling.py                                │
│   modules: sampling/, Protein_Folding/                       │
│   backend: local simulator (default) or IBM QPU              │
│   output:  quantum_data/sampling_result_2/samples/<pdbid>/   │
│            └── samples_<pdbid>_all_ibm.csv                   │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2: Decode + energy + features                          │
│   entry: run_post_process.py                                 │
│   modules: qsadpp/ (orchestrator, decoder, energy, features) │
│   output:  pp_result/<pdbid>/                                │
│            ├── decoded.jsonl                                 │
│            ├── energies.jsonl     ← Stage-3 hand-off         │
│            └── features.jsonl                                │
└──────────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3: argmin(E_total) → CA→full-atom (+ optional polish)  │
│   entry: build_fullatom_results.py                           │
│   modules: post_process/{decom, ca2all, polish}              │
│   output:  qsad_pdb_results/<pdbid>.pdb                      │
│            qsad_pdb_results/build_summary.csv                │
└──────────────────────────────────────────────────────────────┘

qpu_samples/  (recorded 120- and 49-qubit runs on ibm_cleveland)
   │
   ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 4: energy-landscape analysis  →  paper Fig. 9 / Fig. 10│
│   entry: run_landscape.py, plot_landscapes.py                │
│   modules: post_process/landscape/                           │
│   output:  landscape_result/data/<pdbid>_landscape.json      │
│            landscape_result/figures/<pdbid>_landscape.{pdf,png}│
└──────────────────────────────────────────────────────────────┘
```

For each pdbid, Stage 3 picks the `argmin(E_total)` record from
`pp_result/<pdbid>/energies.jsonl`, decodes it into Cα lattice coordinates,
and reconstructs an all-atom coordinate set via NERF backbone with rotamer
side-chain placement (pure Python). An **optional** PyRosetta-based
side-chain polish stage can be enabled at runtime by setting
`QSAD_ENABLE_POLISH=1`; it is **disabled by default**.

### Scope of the reconstruction

The method predicts the **backbone fold**. Stage 3 turns the predicted Cα
trace into an all-atom coordinate set by building each residue with
idealised internal geometry and placing side chains from a rotamer library.
It is a coordinate-generation step, not a physical refinement, and it does
not close the inter-residue peptide bond. For the bundled `1qin` example the
output has

| measurement | value | ideal |
|---|---|---|
| N–Cα | 1.458 Å | 1.458 |
| Cα–C | 1.525 Å | 1.525 |
| N–Cα–C | 111.2° | 111.0 |
| Cα–Cα | 3.800 Å | 3.80 |
| C(i)–N(i+1) | 3.6–4.6 Å | 1.329 |

i.e. textbook intra-residue geometry on a correctly spaced Cα trace, but no
closed peptide bond between residues. The PDB carries no `CONECT` records, so
viewers that infer bonds by distance (PyMOL, ChimeraX) draw the residues as
separate fragments. This is expected. Refining these coordinates into a
physically validated structure is outside the scope of the method.

## Directory layout

```
QSAD/
├── instance_sampling.py       # Stage 1 entry
├── run_post_process.py        # Stage 2 entry
├── build_fullatom_results.py  # Stage 3 entry
├── runtime_paths.py           # project-root + data-root resolver
├── sampling/                  # quantum sampling (ansatz, backends, runner)
├── Protein_Folding/           # MJ Hamiltonian + ProteinFoldingProblem
├── qsadpp/                    # decode / energy / feature orchestrator
├── run_landscape.py           # Stage 4 analysis
├── plot_landscapes.py         # Stage 4 figures (Fig. 9 / Fig. 10)
├── post_process/
│   ├── decom/                 # bitstring → lattice coords
│   ├── rerank/                # candidate model + geometric scoring
│   ├── ca2all/                # NERF backbone + rotamer side-chain
│   ├── polish/                # optional PyRosetta side-chain refinement
│   └── landscape/             # energy-landscape analysis (Stage 4)
├── figs/                      # figure styling used by the paper figures
├── qpu_samples/               # recorded ibm_cleveland runs (1a0q, 4y79)
├── task_input/
│   └── tasks.csv              # input proteins (pdbid + sequence)
├── ibm_config.txt.template    # only needed for live-QPU runs
├── requirements.txt
├── LICENSE                    # Apache License 2.0
├── NOTICE                     # third-party attribution
└── .gitignore
```

## Setup

All commands below assume you are inside the unpacked artifact:

```bash
cd QSAD
```

### 1. Python environment (conda recommended)

```bash
conda create -n qsad python=3.10 -y
conda activate qsad
# Take scipy and scikit-learn from conda: on recent macOS the PyPI wheels
# for Python 3.10 can fail to load (dyld rejects scipy's _propack
# extension with a "zero-fill section type" error).
conda install -c conda-forge scipy scikit-learn -y
pip install -r requirements.txt
```

This is sufficient for the **default** Stage 3 path (NERF + rotamer
side-chain reconstruction, pure Python). `qiskit-ibm-runtime` is imported
only when `QSAD_BACKEND_KIND=ibm`, so the default simulator path does not
depend on the IBM runtime stack being importable.

### 2. (Optional) PyRosetta-based side-chain polish

The polish stage is off by default. Enable it only if needed:

```bash
# extra deps
conda install -c conda-forge openmm pdbfixer -y
# install PyRosetta from https://www.pyrosetta.org/downloads
# (academic licence) into the same conda env

# opt in at runtime
export QSAD_ENABLE_POLISH=1
```

### 3. (Optional) IBM Quantum credentials

Only needed if you want Stage 1 to run on real hardware instead of the
local simulator:

```bash
cp ibm_config.txt.template ibm_config.txt
# Edit ibm_config.txt:
#   TOKEN    = <your IBM Quantum API token>
#   INSTANCE = <your CRN>
#   BACKEND  = ibm_cleveland     # or another backend you have access to
export QSAD_BACKEND_KIND=ibm
```

### 4. `PYTHONPATH`

All entry scripts assume the pipeline root is on `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

## Running the pipeline

### Stage 1 — sampling

Edit `task_input/tasks.csv` to list the proteins you want to fold. Columns:
`pdbid`, `main_chain_residue_seq` (one-letter amino acids). The default
entry is `1qin,QQTMLRV` (7 residues, 33 qubits — small enough to run on
a local Aer simulator as well as the live 127-qubit backend).

```bash
python instance_sampling.py
```

That is the whole of Stage 1. With no `QSAD_BACKEND_KIND` set it runs on
the local `qiskit-aer` simulator, so it needs no IBM account, no token
and no code edit. Per protein it runs 10 ansatz groups of 12 circuits at
2000 shots each — 240,000 shots in total — taking roughly 25 s per group,
i.e. **4–6 min for `1qin`** on a recent laptop. Output lands in
`quantum_data/sampling_result_2/samples/<pdbid>/`. Sampling is
resume-safe: a per-protein `progress.json` under the same run directory
lets a re-launched job pick up unfinished groups.

**Why the simulator can handle 33 qubits.** A dense statevector cannot:
2**33 amplitudes x 16 B = 137 GB of RAM. `sampling/backends.py` therefore
switches to Aer's `matrix_product_state` method for circuits wider than
28 qubits (`QSAD_AER_MPS_THRESHOLD`). MPS is exact and inexpensive for
the QSAD ansatz, which uses linear entanglement with `reps=1`. Force a
different method with `QSAD_AER_METHOD`. Setting `QSAD_SIM_SEED` pins both
the ansatz parameters and the sampling RNG, so two runs with the same seed
produce byte-identical Stage-1 CSVs; leaving it unset draws a fresh random
seed per ansatz group.

**Running on real hardware instead.** Fill in `ibm_config.txt` as in
Setup step 3 and switch the backend:

```bash
QSAD_BACKEND_KIND=ibm python instance_sampling.py
```

Keep separate runs apart with `QSAD_SAMPLING_RUN_DIR`; Stage 2 picks up
whichever run directory it names:

```bash
QSAD_SAMPLING_RUN_DIR=sampling_result_sim python instance_sampling.py
```

### Stage 2 — decode / energy / features

```bash
python run_post_process.py
```

Reads Stage-1 outputs and writes
`pp_result/<pdbid>/{decoded,energies,features}.jsonl`. Cost is linear in
the number of samples: ~3 min for a full 240,000-shot Stage-1 run. The `energies.jsonl`
file carries a 10-term lattice energy decomposition (`E_total`, `E_steric`,
`E_geom`, `E_bond`, `E_mj`, `E_dihedral`, `E_hb`, `E_hydroph`, `E_cbeta`,
`E_rama`) and is the hand-off consumed by Stage 3.

### Stage 3 — full-atom build (optional polish)

```bash
# Default: pure-Python NERF + rotamer reconstruction (no PyRosetta)
python build_fullatom_results.py

# Optional: opt in to PyRosetta side-chain polish
QSAD_ENABLE_POLISH=1 python build_fullatom_results.py
```

For each pdbid under `pp_result/`, Stage 3 selects the lowest-`E_total`
record and produces `qsad_pdb_results/<pdbid>.pdb` together with a
`build_summary.csv` row (columns: `pdbid, sequence, length, best_E_total,
final_energy, status, message, output_pdb`).

### Stage 4 — energy-landscape figures (Fig. 9 and Fig. 10)

The paper's central result is the energy landscape recovered from
utility-scale QPU sampling. Two recorded runs on `ibm_cleveland` ship with the
artifact:

| protein | L | qubits | shots | QPU wall clock |
|---|---|---|---|---|
| `1a0q` | 16 | **120** | 240,000 | 6381 s |
| `4y79` | 9 | 49 | 240,000 | 688 s |

These circuits cannot be simulated classically — a 120-qubit statevector is
out of reach, and matrix-product-state simulation of the 49-qubit circuit
already fails to finish a single 100-shot job in minutes. The recorded samples
are therefore the reproducible starting point; every step after them is
deterministic and runs on a laptop.

```bash
# 4a. analysis: samples → per-protein landscape record
python run_landscape.py

# 4b. figures: landscape → the two paper figures
python plot_landscapes.py
```

`run_landscape.py` decodes every sampled bitstring, keeps the self-avoiding
walks, scores them with the lattice contact potential, projects them onto two
principal components, and writes `landscape_result/data/<pdbid>_landscape.json`
plus `landscape_summary.csv`. Expected values:

| protein | valid SAW conformations | unique bitstrings | funnel score | basins |
|---|---|---|---|---|
| `1a0q` | 266 | 207,324 | 0.745 | 6 |
| `4y79` | 683 | 4,092 | 0.261 | 4 |

`plot_landscapes.py` then writes
`landscape_result/figures/<pdbid>_landscape.{pdf,png}` — `1a0q` is Fig. 9 and
`4y79` is Fig. 10 in the paper. Both take about a minute in total.

The numbers above are exact and version-independent. The figures are
pixel-identical to the published ones under the library versions used for the
paper; a freshly resolved environment gives visually identical figures with
sub-pixel rendering differences (under 0.5% of pixels) from `matplotlib` and
`numpy` patch releases.

## Environment variables

| var | purpose | default |
|---|---|---|
| `QSAD_DATA_ROOT` | absolute path for sampling / post-processing I/O | `<root>/quantum_data` |
| `QSAD_SAMPLING_RUN_DIR` | subdir under data root for this run | `sampling_result_2` |
| `QSAD_POSTPROCESS_INPUT_DIR` | override input root for Stage 2 | auto-resolved |
| `QSAD_ENABLE_POLISH` | turn on the optional PyRosetta polish in Stage 3 | `0` (off) |
| `QSAD_BACKEND_KIND` | Stage-1 backend: `simulator` (local) or `ibm` (live QPU) | `simulator` |
| `QSAD_AER_METHOD` | force an Aer simulation method (e.g. `matrix_product_state`) | auto by circuit width |
| `QSAD_AER_MPS_THRESHOLD` | qubit width above which Aer switches to MPS | `28` |
| `QSAD_SIM_SEED` | pins ansatz parameters + sampling RNG for a byte-reproducible run | unset (random per group) |
| `IBM_CONFIG_FILE` | IBM credential file path | `./ibm_config.txt` |
| `TASKS_CSV` | input task list | `./task_input/tasks.csv` |

## License

QSAD is released under the Apache License, Version 2.0 — see [LICENSE](LICENSE).

`Protein_Folding/` contains code derived from the protein-folding problem
implementation of Qiskit Nature, (C) Copyright IBM 2021, 2022, also under
Apache-2.0. The original copyright headers are retained in those files, and
[NOTICE](NOTICE) records the attribution.

## Citation

If you use QSAD in your research, please cite:

```bibtex
@article{zhang2026quantum,
  title={Quantum Sampling Architecture for Protein Structure Reconstruction on Utility-Scale Hardware},
  author={Zhang, Yuqi and Fang, Bo and Yang, Yuxin and Cheng, Feixiong and Chen, Jieyang and Fang, Sherry and Chen, Siwei and Zhao, Junhan and Guan, Qiang},
  journal={arXiv preprint arXiv:2607.06971},
  year={2026}
}
```
