"""Generic utilities: seeding, logging, EMA, checkpointing, meters."""
from __future__ import annotations

import copy
import glob
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    # Re-point the file handler when a DIFFERENT log_file is requested for
    # an already-configured logger (fixed 2026-09-15, real bug observed on
    # cluster job 6439). `logging.getLogger(name)` returns a process-wide
    # SINGLETON, and the previous `if logger.handlers: return logger`
    # short-circuit meant the second and later calls kept the FIRST call's
    # file handler. scripts/run_ablation_matrix.py calls train() once per
    # module combo in a single process, so every combo after the first
    # wrote its training log into the FIRST combo's train.log -- job 6439's
    # runs/ablation_collective_v3/fc/train.log ended up containing
    # fc+aep+rv's output (disentangle/reverse/rv_calibration losses and
    # verification_auroc, none of which exist for an fc-only run), making
    # the per-combo logs actively misleading. Checkpoints were unaffected
    # (they use cfg.run.output_dir directly, not the logger), so no result
    # was corrupted -- but the logs could not be trusted to describe the
    # run whose directory they sat in.
    existing_files = {getattr(h, "baseFilename", None) for h in logger.handlers}
    wanted = os.path.abspath(log_file) if log_file else None
    if logger.handlers and (wanted is None or wanted in existing_files):
        return logger
    if logger.handlers:
        # A different file was asked for: drop the stale handlers (closing
        # file handles so the old log is flushed) and rebuild below.
        for h in list(logger.handlers):
            h.close()
            logger.removeHandler(h)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class AverageMeter:
    """Tracks a running mean; used for loss/metric logging."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class EMA:
    """Exponential moving average of model weights (optim.ema_decay > 0 enables)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


class EarlyStopper:
    """Generic early-stopping tracker driven by config.early_stopping.

    Monitors a scalar metric; stops when it hasn't improved by at least
    `min_delta` for `patience` consecutive evaluation rounds, after an
    initial `warmup_epochs` grace period (so a lucky/unlucky first epoch
    cannot trigger or block stopping).
    """

    def __init__(self, mode: str = "max", patience: int = 5, min_delta: float = 1e-4, warmup_epochs: int = 1):
        assert mode in ("max", "min")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best = -float("inf") if mode == "max" else float("inf")
        self.num_bad_epochs = 0
        self.best_epoch = -1

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> Dict[str, Any]:
        improved = self._is_improvement(value)
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        should_stop = (epoch >= self.warmup_epochs) and (self.num_bad_epochs >= self.patience)
        return {"improved": improved, "should_stop": should_stop, "best": self.best, "best_epoch": self.best_epoch,
                "num_bad_epochs": self.num_bad_epochs}


def _trainable_state_dict(model: torch.nn.Module, trainable_names: Optional[set] = None) -> dict:
    """`state_dict()` filtered to trainable parameters, plus every buffer, plus
    frozen parameters that are NOT safely reconstructible without the
    checkpoint.

    Only `backbone.clip.*` (HuggingFace's pretrained CLIP weights) is treated
    as safely excludable when frozen: reconstructing `HFClipBackbone` downloads
    the same pretrained checkpoint every time, so those specific frozen values
    are deterministic and dropping them from `.pt` files is free (they were
    1.71 GB of a 2.0 GB checkpoint here). This is why the exclusion is a name
    prefix, not a blanket "any frozen param" rule: `TinyBackbone` (the offline
    smoke-test fallback) also freezes parameters when `freeze_backbone: true`,
    but at least one of them (`text_pos`) is `nn.Parameter(torch.randn(...))` --
    random at construction, never loaded from anywhere -- so reconstructing the
    model gives a DIFFERENT value and excluding it would silently corrupt
    resume. A frozen-but-random parameter is not the same thing as a
    frozen-and-pretrained one, and only the checkpoint file can tell those
    apart; a blanket `requires_grad` filter cannot.

    `trainable_names` MUST be passed explicitly when filtering an EMA shadow:
    `EMA.__init__` calls `requires_grad_(False)` on every shadow parameter
    (it is only ever updated under `torch.no_grad()`), so deriving "trainable"
    from the shadow's own `requires_grad` would exclude everything -- not just
    the frozen backbone. Compute the name-set once from the LIVE model and
    reuse it for both `live_model` and `ema.shadow`.
    """
    if trainable_names is None:
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    param_names = dict(model.named_parameters())

    def excludable(name: str) -> bool:
        return name.startswith("backbone.clip.") and name not in trainable_names

    full = model.state_dict()
    return {k: v for k, v in full.items() if k not in param_names or k in trainable_names or not excludable(k)}


def capture_rng_state() -> dict:
    """Snapshots every RNG stream training draws from -- Python's `random`
    (used by some dataset-side augmentation/sampling code), NumPy (ditto),
    PyTorch CPU, and PyTorch CUDA (added 2026-09-09, code-audit finding #9).

    Per-epoch DataLoader seeding (`train.py`'s `make_train_loader`, seeded
    from `cfg.run.seed + epoch`) already makes sample ORDER exactly
    reproducible across a resume on its own -- that mechanism is independent
    of this one and unaffected by it. What per-epoch seeding does NOT cover
    is everything else that consumes these same global RNG streams
    mid-training: dropout masks, any stochastic CUDA kernel, or future
    random model-side operations. Without restoring these streams to
    exactly where an uninterrupted run would have left them, a resumed run's
    dropout masks (etc.) diverge from what continuous training would have
    drawn from the very next step onward -- not incorrect, but not an EXACT
    resume either, which matters for a reproducibility claim specifically
    (see restore_rng_state's docstring for where this is actually used).
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state: Optional[dict]) -> None:
    """Restores every RNG stream `capture_rng_state()` snapshotted (added
    2026-09-09, code-audit finding #9). Call once, immediately after
    `load_checkpoint()`, before continuing training -- train.py's resume
    block does this right after loading the checkpoint dict.

    A checkpoint saved before this fix existed has no `"rng_state"` key;
    `rng_state=None` is a deliberate, silent no-op (old checkpoints must
    still resume, just without the exact-RNG guarantee they never had in
    the first place -- this is optional reproducibility polish, not a
    correctness requirement, per the code-audit document's own "not
    required for the next ordinary training run" priority note).
    """
    if rng_state is None:
        return
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    # `.cpu()`: train.py's resume loads the checkpoint with map_location=device,
    # which moves these ByteTensors to the GPU too, and both setters accept only
    # a CPU ByteTensor ("RNG state must be a torch.ByteTensor", seed-study
    # resume job 7745, 2026-10-26). The bytes, and so the restored state, are unchanged.
    torch.set_rng_state(rng_state["torch"].cpu())
    if rng_state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng_state["cuda"]])


def save_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None,
                     epoch: int = 0, best_metric: float = 0.0, config: Optional[dict] = None,
                     extra: Optional[dict] = None, ema=None, live_model: Optional[torch.nn.Module] = None,
                     scaler=None, step_in_epoch: Optional[int] = None) -> None:
    """`model` is whichever model was just evaluated (the EMA shadow when EMA is
    enabled, else the live training model -- see train.py's `eval_model`).

    Pass `ema` (the EMA wrapper) and `live_model` (the actual training model,
    which is what `optimizer`'s state belongs to) whenever EMA is in use. Without
    them, a resume that loads `state["model"]` into the live model pairs EMA
    weights with optimizer momentum that was tracking a *different* set of
    weights the whole time, and a fresh EMA shadow starts averaging from
    scratch -- this reset a real run's validation R@1 from 13.25 back to ~7 on
    resume before it was caught.

    Pass `scaler` (the `torch.amp.GradScaler`) so its dynamic loss-scale value
    survives resume too -- otherwise every resume restarts AMP's scale
    calibration from scratch, risking a few skipped/unstable steps right after
    every reconnect.

    Pass `step_in_epoch` for a mid-epoch checkpoint (see `train.py`'s periodic
    save): `epoch` then means "the epoch in progress", and resume replays the
    DataLoader to `step_in_epoch` before continuing, rather than skipping the
    epoch's remaining steps entirely.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Derive the trainable-name set from the LIVE model specifically: the EMA
    # shadow (utils.py:EMA.__init__) sets requires_grad=False on every one of
    # its own parameters, so asking `model` for this set would silently
    # produce an empty trainable set whenever `model` is the shadow (the usual
    # case -- `model` is "whichever model was just evaluated", which is the
    # shadow when EMA is enabled). live_model, when passed, is always the real
    # training model and is the correct source of truth; fall back to `model`
    # only when there is no separate live model (EMA disabled).
    trainable_names = {n for n, p in (live_model or model).named_parameters() if p.requires_grad}
    state = {
        "model": _trainable_state_dict(model, trainable_names),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "best_metric": best_metric,
        "config": config,
        "extra": extra or {},
        "live_model": _trainable_state_dict(live_model, trainable_names) if live_model is not None else None,
        "ema_shadow": _trainable_state_dict(ema.shadow, trainable_names) if ema is not None else None,
        "rng_state": capture_rng_state(),
    }
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None, map_location="cpu",
                     ema=None, scaler=None) -> dict:
    """Loads the LIVE model (paired with the saved optimizer state) into `model`.

    Falls back to `state["model"]` for old checkpoints saved before `live_model`
    existed -- those were correct only when EMA was disabled; log a warning
    otherwise so a stale-checkpoint resume isn't silently wrong again.

    Checkpoints only contain TRAINABLE parameters (see `_trainable_state_dict`
    in `save_checkpoint`) -- `strict=False` is required, not a safety net: it is
    how frozen backbone weights are meant to come from `model`'s own
    construction (the same pretrained CLIP checkpoint every time) rather than
    from disk.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    live_sd = state.get("live_model") or state["model"]
    if state.get("live_model") is None and state.get("ema_shadow") is None and ema is not None:
        import warnings
        warnings.warn(
            f"{path} predates checkpointing EMA separately from the live model. "
            "Loading state['model'] (the EMA shadow at save time) into the live "
            "training model -- optimizer momentum will not match these weights. "
            "This is the exact mismatch that reset a training run's R@1 from "
            "13.25 to ~7 on resume; expect a similar dip here.")
    # Resize any LAZILY-SHAPED buffers (e.g. CrossBatchMemory's pooled_queue/
    # tokens_queue in reverse_verifier.py, which start as empty torch.zeros(0)
    # placeholders because their real shape -- backbone embed_dim / patch
    # count -- isn't known until the first real forward pass) to match the
    # checkpoint's shape before loading. Without this, resuming a run whose
    # queue was already populated raises a hard size-mismatch error, because
    # a freshly-constructed model's buffer is still the (0,)-shaped
    # placeholder. This only ever fires for genuine placeholder buffers
    # (0-numel on the live model) -- a real shape mismatch elsewhere (e.g. a
    # config change between save and resume) still raises normally via
    # load_state_dict below, since this loop only rewrites tensors that are
    # currently empty.
    live_model_sd = model.state_dict()
    for name, ckpt_tensor in live_sd.items():
        live_tensor = live_model_sd.get(name)
        if (live_tensor is not None and live_tensor.numel() == 0
                and torch.is_tensor(ckpt_tensor) and ckpt_tensor.numel() > 0):
            module_path, _, buf_name = name.rpartition(".")
            submodule = model.get_submodule(module_path) if module_path else model
            submodule.register_buffer(buf_name, torch.zeros_like(ckpt_tensor))
            if hasattr(submodule, "_shape_initialized"):
                submodule._shape_initialized = True

    missing, unexpected = model.load_state_dict(live_sd, strict=False)
    # Fixed 2026-09-08 (real bug found via code-audit document review):
    # `model.state_dict()[n].requires_grad` is ALWAYS False -- state_dict()
    # returns detached tensor copies regardless of the source parameter's
    # actual trainability (confirmed directly: a plain nn.Linear's weight
    # has requires_grad=True but its state_dict() entry reports False). This
    # made `non_trainable_missing` an ALWAYS-EMPTY list, silently defeating
    # the warning's entire purpose -- it could never fire, so a missing
    # trainable parameter after an architecture change (e.g. this session's
    # own residual_alpha/rv_calibration additions) would resume silently at
    # its default init with no warning at all. Fixed by checking
    # `model.named_parameters()` instead, which correctly reports
    # requires_grad on the LIVE parameter, not a detached copy of it.
    trainable_param_names = {n for n, p in model.named_parameters() if p.requires_grad}
    non_trainable_missing = [n for n in missing if n in trainable_param_names]
    if non_trainable_missing:
        import warnings
        warnings.warn(f"{path}: {len(non_trainable_missing)} TRAINABLE params missing from checkpoint "
                      f"(not just frozen ones) -- e.g. {non_trainable_missing[:3]}. Resume may be incomplete.")
    if unexpected:
        import warnings
        warnings.warn(f"{path}: {len(unexpected)} unexpected keys in checkpoint, ignored: {unexpected[:3]}")
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if ema is not None and state.get("ema_shadow") is not None:
        ema.shadow.load_state_dict(state["ema_shadow"], strict=False)
    return state


class TopKCheckpointKeeper:
    """Keeps only the K best checkpoints on disk (Section: implementation detail, save_top_k)."""

    def __init__(self, k: int = 3, mode: str = "max"):
        self.k = k
        self.mode = mode
        self._entries = []  # list of (metric, path)

    def offer(self, metric: float, path: str) -> None:
        self._entries.append((metric, path))
        reverse = self.mode == "max"
        self._entries.sort(key=lambda x: x[0], reverse=reverse)
        while len(self._entries) > self.k:
            _, worst_path = self._entries.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)

    def rebuild_from_disk(self, output_dir: str) -> None:
        """Reconstructs `_entries` from whatever `epochNNN.pt` checkpoints
        already exist in `output_dir` (added 2026-09-09, code-audit finding
        #8): a freshly-constructed keeper starts with an empty `_entries`
        list regardless of whether this is a from-scratch run or a resume of
        an interrupted one. On resume, prior epochs' checkpoints are already
        on disk but the new keeper instance has no memory of them -- its own
        `k`-file retention logic (`offer()`'s `while len(self._entries) >
        self.k: ... os.remove(worst_path)`) then only ever prunes checkpoints
        saved AFTER the resume, so a run interrupted and resumed multiple
        times can accumulate more than `save_top_k` files on disk (wasted
        space, and checkpoint retention that's inconsistent between an
        uninterrupted run and an interrupted-then-resumed one covering the
        same epochs).

        Reconstructs by SCANNING disk (Option B from the code-audit
        document, preferred over persisting separate keeper state in each
        checkpoint's `extra` dict): each `epochNNN.pt` already stores its own
        `best_metric` under the same key `save_checkpoint()` writes on every
        save, so the keeper's state is fully recoverable from the files
        themselves with no new checkpoint fields and no risk of the
        persisted-state copy drifting from what's actually on disk (e.g. if
        a checkpoint file is manually deleted/moved between runs, a
        persisted-state file would silently claim it still exists; a disk
        scan cannot make that mistake). Call once, right after construction,
        before the training loop -- calling it after `offer()` has already
        been called this run would incorrectly re-merge already-tracked
        entries. On a from-scratch run (no existing `epochNNN.pt` files),
        this is a no-op.
        """
        pattern = os.path.join(output_dir, "epoch*.pt")
        for path in sorted(glob.glob(pattern)):
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue  # corrupt/partially-written file -- ignore, don't crash resume over it
            metric = state.get("best_metric")
            if metric is not None:
                self._entries.append((float(metric), path))
        reverse = self.mode == "max"
        self._entries.sort(key=lambda x: x[0], reverse=reverse)
        # Note: does NOT prune here even if more than `k` files are found --
        # a resume should not silently delete checkpoints from a
        # differently-configured earlier run (e.g. save_top_k was lowered
        # between runs); pruning only happens going forward, the next time
        # offer() pushes the tracked count over `self.k`.
