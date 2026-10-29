"""Upload a local folder to a PRIVATE Hugging Face repo (token from HF_TOKEN in .env).

  venv/bin/python scripts/hf_upload.py work/asr/phowhisper --repo <user>/<name> --type dataset --dry-run
  venv/bin/python scripts/hf_upload.py work/models/phowhisper-large-ct2 --repo <user>/phowhisper-large-ct2 --type model

The competition licence forbids redistributing the data or anything derived from it per sample (ASR output, labels), so
the repo is always created private and an already-public target is refused.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--repo", required=True, help="<user>/<name>")
    ap.add_argument("--type", default="model", choices=["model", "dataset"])
    ap.add_argument("--dry-run", action="store_true", help="list the files and sizes, upload nothing")
    a = ap.parse_args()
    load_env()
    if not a.folder.is_dir():
        sys.exit(f"not a folder: {a.folder}")
    files = [p for p in a.folder.rglob("*") if p.is_file()]
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB from {a.folder}")
    if a.dry_run:
        return
    api = HfApi(token=os.environ["HF_TOKEN"])
    print("logged in as", api.whoami()["name"])
    try:
        if not api.repo_info(a.repo, repo_type=a.type).private:
            sys.exit(f"{a.repo} is public: refusing to upload")
    except Exception as e:  # not found -> create it private
        if "404" not in str(e) and "not found" not in str(e).lower():
            raise
    api.create_repo(a.repo, repo_type=a.type, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(a.folder), repo_id=a.repo, repo_type=a.type)
    print(f"uploaded to https://huggingface.co/{a.repo} (private)")


if __name__ == "__main__":
    main()
