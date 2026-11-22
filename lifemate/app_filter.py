"""App filter: keep the K installed apps whose description is closest to what the user said (fine-tuned e5).

Model: intfloat/multilingual-e5-small fine-tuned on train folds 1-4 (scripts/train_app_embedder.py, Kaggle job
quyenuit24/lifemate-embed-train; local copy work/kaggle/embed_train_q/out/app_embedder_e5). Query and app documents are
built exactly as in training: "query: " + normalize_text(final user text), "passage: " + enriched_app_doc(app).
An earlier fold-0 check recorded 100% sample recall, but that is not a guarantee of top-7. The paired 309-sample
RAT checkpoint evaluation retained 455/498 required gold-app instances (91.37%) and all required apps in 244/309
samples. Recall depends on the query and target definition; measure it on the exact evaluation set before relying on
the filter.

    f = AppFilter(model_dir)                # once: loads e5 and embeds every catalog app
    keep = f.keep(text, installed_apps)     # -> set of app ids for run_sample(keep_apps=...)

`contacts` is always kept when installed; run_sample always keeps sys.*.
"""
from __future__ import annotations

from pathlib import Path

from lifemate.data import DATA_ROOT, load_catalog
from lifemate.normalize import normalize_text

TOP_K = 7
ALWAYS = ("contacts",)


def app_doc(a: dict) -> str:
    """Same text as scripts/train_app_embedder.enriched_app_doc (the model was trained on it)."""
    app_id = a["id"].replace("_", " ")
    tools = "; ".join(f"{t['name'].split('.')[-1].replace('_', ' ')}: {t.get('description', '')}" for t in a.get("tools", []))
    return f"{a.get('name_vi', '')} ({app_id}): {a.get('description', '')}. Các chức năng: {tools}"


class AppFilter:
    def __init__(self, model_dir: str | Path, split: str = "train", root: Path = DATA_ROOT, k: int = TOP_K,
                 device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.k = torch, k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir)).to(self.device).eval()
        apps = load_catalog(split, root)["apps"]
        self.ids = [a["id"] for a in apps]
        self.docs = dict(zip(self.ids, self._embed([f"passage: {app_doc(a)}" for a in apps], 512)))

    def _embed(self, texts: list[str], max_len: int):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                b = self.tok(texts[i:i + 32], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
                h = self.model(**b)[0]
                m = b["attention_mask"].unsqueeze(-1).float()
                out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1).cpu())
        return torch.cat(out)

    def scores(self, text: str, installed: list[str]) -> dict[str, float]:
        q = self._embed([f"query: {normalize_text(text)}"], 128)[0]
        return {a: float(self.docs[a] @ q) for a in installed if a in self.docs}

    def keep(self, text: str, installed: list[str]) -> set[str]:
        s = self.scores(text, installed)
        kept = set(sorted(s, key=lambda a: -s[a])[: self.k])
        return kept | {a for a in ALWAYS if a in installed}
