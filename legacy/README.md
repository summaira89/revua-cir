# Legacy files

Kept for provenance only. Nothing here produces a number reported in the paper, and nothing
here is needed to reproduce it.

- `scripts/`: one-off diagnostics written while debugging earlier versions of the pipeline
  (reranking-window and parity checks, image-path probes, a cluster-to-laptop data-transfer
  helper, a checkpoint backup helper and an early figure script). Some refer to files or
  folders that are not part of this repository.
- `tests/`: tests of the transfer helper and of a cluster-only shell script.
- `configs/`: configurations of superseded training attempts (frozen or partially frozen
  backbones, small batches, short verification runs). The reported checkpoints were trained
  with `configs/stage1_finetune_full_unfreeze_gradcache.yaml`; see the README.
