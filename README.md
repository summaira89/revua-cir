# ReVUA-CIR: auditing learned verification in composed image retrieval

Code accompanying the paper *Auditing Learned Verification in Composed Image Retrieval:
Candidate Selection, Simple Evidence and Transfer* (S. Nosheen, G. Mustafa; submitted to
*Image and Vision Computing*).

The paper asks whether a learned second-stage verifier adds evidence beyond inexpensive
similarities. It compares second-stage evidence (learned verifier, text–image cosine,
pretrained CLIP) on matched top-10 candidate windows from two first-stage retrievers
(ReVUA-CIR and CLIP4Cir), with identical fusion and tuning, paired significance tests and a
frozen-weight transfer test to CIRCO.

## Repository layout

| Path | Contents |
|---|---|
| `revua_cir/` | Model (forward composer, attribute edit parser, reverse verifier), training, evaluation, metrics, significance tests |
| `configs/` | Training and evaluation configs |
| `scripts/` | Evaluation, cache building and every analysis in the paper (see below) |
| `tests/` | Unit tests (`python tests/run_all_tests.py`) |
| `results/` | Locked result files used in the paper, and their SHA-256 lock files |
| `legacy/` | Development-only diagnostics and superseded configs, kept for provenance (see `legacy/README.md`) |

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10–3.12, PyTorch 2.1+ and CUDA 12.

## Data and weights

Datasets are not redistributed; download them from their providers:

- CIRR: https://github.com/Cuberick-Orion/CIRR
- CIRCO: https://github.com/miccunifi/CIRCO
- FashionIQ: https://github.com/XiaoxiaoGuo/fashion-iq
- CLIP4Cir (external retriever, released RN50x4 weights): https://github.com/ABaldrati/CLIP4Cir

Per-query score caches, candidate windows (`.npz`) and the trained checkpoint are archived
on Zenodo: **DOI to be added**.

## Reproducing the paper

| Paper item | Script |
|---|---|
| Train the model | `python -m revua_cir.train --config configs/stage1_finetune_full_unfreeze_gradcache.yaml --data-root ./datasets` (the reported checkpoints; `scripts/run_ablation_matrix.py` trains the `FC` and `FC+AEP+RV` pair under one protocol) |
| CIRR / CIRCO evaluation | `scripts/evaluate_cirr.py`, `scripts/evaluate_circo.py` |
| Encoder towers (text cosine, pretrained CLIP) | `scripts/encode_towers.py`, `scripts/circo_towers.py` |
| CLIP4Cir windows | `scripts/clip4cir_probe.py`, `scripts/cirr_external_windows.py` |
| Q1–Q4 analyses and supporting studies | `scripts/revision_followups*.py` |
| Figures and graphical abstract | `scripts/make_ivc_figures.py` |

The reported runs used `configs/stage1_finetune_full_unfreeze_gradcache.yaml` with early-stopping patience 5 after 2 warm-up epochs (the file now reads 3 and 1, a later change). The settings stored in each checkpoint are authoritative and are listed in the paper's supplementary training table.

The analyses run on CPU from the cached windows; only training and encoding need a GPU.
Each script's docstring gives its inputs and command line. All data paths default to
`./datasets` and can be changed with `--data-root` (or `data.root` in the config).

No script contacts an evaluation server. `scripts/cirr_test.py` and `scripts/evaluate_cirr.py`
only write a CIRR test-split prediction file to disk; submitting it to the CIRR server is a
manual step. The EDIR scripts run only when invoked explicitly, and the reported EDIR study
was a single pre-registered run that should not be repeated to tune anything.

## Verifying the locked results

`results/locks/*.sha256` record the hashes of the result files reported in the paper. Paths
inside them are relative to the original project root (`revision_work/...`); here those
files are under `results/revision_work/`. To check them:

```bash
cd results && sha256sum -c locks/LOCK_followups28_results_2026-11-05.sha256
```

Locks that refer to `.npz` windows need the Zenodo archive.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code.

## Licence

MIT, see `LICENSE`.
