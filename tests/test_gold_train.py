"""Integration checks on the real train split (skipped when the data package is not present)."""
import pytest

from lifemate.data import DATA_ROOT, fuzzy_params
from lifemate.gold import build_gold, initial_hash
from lifemate.scoring import score_split

pytestmark = pytest.mark.skipif(not (DATA_ROOT / "train" / "train.jsonl").is_file(), reason="data package not present")


@pytest.fixture(scope="module")
def gold():
    return build_gold("train")


def test_every_gold_replays_exactly(gold):
    assert [sid for sid, g in gold.items() if not g["replay_ok"]] == []


def test_gold_scores_100(gold):
    final, _ = score_split(gold, gold, fuzzy_params("train"))
    assert final == 100.0


def test_empty_submission_floor(gold):
    preds = {sid: {"tool_calls": [], "state_hash": initial_hash("train", sid)} for sid in gold}
    final, per = score_split(preds, gold, fuzzy_params("train"))
    n_empty = sum(1 for g in gold.values() if not g["tool_calls"])
    assert final == pytest.approx(100.0 * n_empty / len(gold))
