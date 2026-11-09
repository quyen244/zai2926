"""Push / cancel a Kaggle GPU job, keeping the session id the CLI does not print (needed to cancel a run).

  python scripts/kaggle_job.py push kaggle/plan_sft_train          # push with GPU T4 x2, save the session id
  python scripts/kaggle_job.py cancel kaggle/plan_sft_train        # cancel the session saved by the last push
  python scripts/kaggle_job.py status kaggle/plan_sft_train

The session id goes to work/kaggle/sessions/<kernel slug>.json (git-ignored). Credentials: the usual ~/.kaggle/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lifemate.data import PROJECT  # noqa: E402

SESSIONS = PROJECT / "work" / "kaggle" / "sessions"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "cancel", "status"])
    ap.add_argument("folder", type=Path, help="job folder with kernel-metadata.json")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    a = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    meta = json.loads((a.folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    ref = meta["id"]
    saved = SESSIONS / f"{ref.split('/')[-1]}.json"
    if a.action == "push":
        r = api.kernels_push(str(a.folder), acc=a.accelerator)
        if getattr(r, "error", None):
            raise SystemExit(f"push failed: {r.error}")
        # NOTE: the push already happened when anything below fails; do not push again to "retry"
        SESSIONS.mkdir(parents=True, exist_ok=True)
        d = r.to_dict() if hasattr(r, "to_dict") else vars(r)  # field names differ between kaggle CLI versions
        pick = lambda *ks: next((d[k] for k in ks if d.get(k) not in (None, "")), None)  # noqa: E731
        info = {"ref": ref, "version": pick("versionNumber", "version_number"),
                "session": pick("kernelSessionId", "kernel_session_id"), "url": pick("url"), "raw": d}
        saved.write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"pushed {ref} version {info['version']}, session {info['session']} -> {saved}")
    elif a.action == "cancel":
        info = json.loads(saved.read_text(encoding="utf-8"))
        from kagglesdk.kernels.types.kernels_api_service import ApiCancelKernelSessionRequest

        req = ApiCancelKernelSessionRequest()
        req.kernel_session_id = int(info["session"])
        with api.build_kaggle_client() as client:
            client.kernels.kernels_api_client.cancel_kernel_session(req)
        print(f"cancel sent for {ref} session {info['session']}")
    print(f"status: {api.kernels_status(ref).status}")


if __name__ == "__main__":
    main()
