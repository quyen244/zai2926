"""Package the RAT SFT rows with tool-choice rationales only for Kaggle."""
import io, json, shutil, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/kaggle_dataset_rat_toolonly"
DATA = ROOT / "work/sft_rat_toolonly"
cv = json.loads((ROOT / "splits/cv5.json").read_text(encoding="utf-8"))
train = [x for x in (DATA / "traj.train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
val = [x for x in (DATA / "traj.val.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
ids = {json.loads(x).get("source_id", json.loads(x)["id"]) for x in train}
if any(cv.get(i) == 0 for i in ids) or ids & {json.loads(x)["id"] for x in val}:
    raise SystemExit("training rows overlap CV fold 0 or validation rows")
calls = [m for x in train for m in json.loads(x)["messages"] if m.get("train") and m.get("tool_calls")]
if not calls or any("<select>" not in (m.get("content") or "") or "<args>" in (m.get("content") or "") for m in calls):
    raise SystemExit("expected every supervised call to have tool-selection reasoning only")
OUT.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUT / "adapter", ignore_errors=True)
shutil.copytree(ROOT / "submission_docker/build/code/saved_models/rec_adapter", OUT / "adapter", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
with tarfile.open(OUT / "repo.tar.gz", "w:gz") as tar:
    tar.add(ROOT / "lifemate", arcname="lifemate", filter=lambda m: None if "__pycache__" in m.name or m.name.endswith(".pyc") else m)
    for name, rows in (("traj.train.jsonl", train), ("traj.val.jsonl", val)):
        data = ("\n".join(rows) + "\n").encode()
        info = tarfile.TarInfo(f"work/sft_rat_toolonly/{name}"); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
(OUT / "dataset-metadata.json").write_text(json.dumps({"title":"lifemate-rat-toolonly-data","id":"quyenuit24/lifemate-rat-toolonly-data","licenses":[{"name":"other"}]}, indent=2), encoding="utf-8")
print(f"packed {len(train)} train / {len(val)} val rows; {len(calls)} rationale calls; {OUT / 'repo.tar.gz'}")
