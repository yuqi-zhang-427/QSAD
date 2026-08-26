# QSAD Pipeline

Quantum-sampled protein folding pipeline — from quantum bitstring
sampling to an all-atom PDB coordinate set. The prediction target is the
protein **backbone**; see [Scope of the reconstruction](#scope-of-the-reconstruction).

This is a **code-only** artifact: no sampling results are bundled. Every
file the pipeline consumes downstream is produced by running Stage 1,
which by default executes on the local `qiskit-aer` simulator and needs
no credentials of any kind.

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
├── post_process/
│   ├── decom/                 # bitstring → lattice coords
│   ├── rerank/                # candidate model + geometric scoring
│   ├── ca2all/                # NERF backbone + rotamer side-chain
│   └── polish/                # optional PyRosetta side-chain refinement
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
