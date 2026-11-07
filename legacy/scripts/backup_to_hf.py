"""Move checkpoint files from the cluster to a PRIVATE Hugging Face repository, verified.

For each file:
  1. compute its local SHA-256;
  2. upload it to a private model repository (created if absent), under
     the same relative path it has below --base;
  3. read back the SHA-256 that Hugging Face itself recorded for the
     uploaded (LFS) object and require it to equal the local one;
  4. only with --delete-after-verify, and only if steps 1-3 passed for
     that file, delete the local copy.
A manifest of path, size and SHA-256 is uploaded to the repository too.

The token is read from the environment (HF_TOKEN) and is never written to
disk or printed. Create one at https://huggingface.co/settings/tokens with
"write" permission.

Usage (on the cluster login node)
-----
  read -s HF_TOKEN && export HF_TOKEN
  python scripts/backup_to_hf.py --repo YOUR_HF_USERNAME/revua-cir-checkpoints \\
      --base ./runs \\
      stage1_finetune_full_unfreeze_gradcache_v3/epoch005.pt b64_retrain/fc_aep/epoch003.pt
  # after reading the "VERIFIED" lines, run again with --delete-after-verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Optional, Sequence


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def remote_sha256(api, repo: str, path: str) -> Optional[str]:
    """The SHA-256 Hugging Face recorded for an uploaded LFS file (None if absent)."""
    info = api.get_paths_info(repo, [path], expand=True, repo_type="model")
    if not info:
        return None
    lfs = getattr(info[0], "lfs", None)
    return getattr(lfs, "sha256", None) if lfs is not None else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="USERNAME/REPO_NAME (created private if absent)")
    ap.add_argument("--base", required=True, help="directory the file paths are relative to")
    ap.add_argument("files", nargs="+", help="file paths relative to --base")
    ap.add_argument("--delete-after-verify", action="store_true",
                    help="delete each local file whose upload is verified (irreversible)")
    args = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        ap.error("set HF_TOKEN in the environment first (read -s HF_TOKEN && export HF_TOKEN)")
    from huggingface_hub import HfApi

    base = os.path.abspath(os.path.expanduser(args.base))
    files = [os.path.normpath(f).replace(os.sep, "/") for f in args.files]
    for f in files:
        if f.startswith("..") or os.path.isabs(f) or not os.path.isfile(os.path.join(base, f)):
            ap.error("%s is not a file below %s" % (f, base))
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(args.repo).private:
        sys.exit("FATAL: %s exists and is PUBLIC; refusing to upload checkpoints to it" % args.repo)
    print("repository %s (private)" % args.repo, flush=True)

    manifest, verified = [], []
    for f in files:
        local = os.path.join(base, f)
        size = os.path.getsize(local)
        t0 = time.time()
        sha = sha256_file(local)
        print("[%s] %.2f GB, local sha256 %s (%.0fs)" % (f, size / 1e9, sha, time.time() - t0), flush=True)
        if remote_sha256(api, args.repo, f) == sha:
            print("[%s] already on Hugging Face with the same sha256; not re-uploaded" % f, flush=True)
        else:
            t0 = time.time()
            api.upload_file(path_or_fileobj=local, path_in_repo=f, repo_id=args.repo, repo_type="model",
                            commit_message="backup %s (sha256 %s)" % (f, sha))
            print("[%s] uploaded in %.0f min" % (f, (time.time() - t0) / 60), flush=True)
        remote = remote_sha256(api, args.repo, f)
        if remote != sha:
            print("[%s] NOT VERIFIED: remote sha256 %s != local %s -- local file kept" % (f, remote, sha), flush=True)
            continue
        print("[%s] VERIFIED: Hugging Face sha256 equals the local sha256" % f, flush=True)
        verified.append(f)
        manifest.append({"path": f, "bytes": size, "sha256": sha})

    if manifest:
        api.upload_file(path_or_fileobj=json.dumps(manifest, indent=2).encode("utf-8"),
                        path_in_repo="backup_manifest_%s.json" % time.strftime("%Y%m%d_%H%M%S"),
                        repo_id=args.repo, repo_type="model", commit_message="backup manifest")
    for f in verified:
        if args.delete_after_verify:
            os.remove(os.path.join(base, f))
            print("[%s] local copy deleted" % f, flush=True)
    print("verified %d of %d file(s)%s" % (len(verified), len(files),
                                         "; local copies deleted" if args.delete_after_verify else
                                         "; nothing deleted (run again with --delete-after-verify)"), flush=True)
    return 0 if len(verified) == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
