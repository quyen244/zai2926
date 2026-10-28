from lifemate.scoring import calls_match, max_matching, score_sample, token_f1

FUZZY = {"ride_hailing.book": {"pickup", "dropoff"}, "sms.send": {"content"}}
H = "h" * 64

GOLD = [
    {"name": "smart_home.get_status", "arguments": {}},
    {"name": "smart_home.climate_set", "arguments": {"device_id": "ac_living_room", "mode": "cool", "temperature": 26}},
    {"name": "ride_hailing.book", "arguments": {"pickup": "227 Nguyễn Văn Cừ, Quận 5", "dropoff": "123 Trần Hưng Đạo, Quận 5"}},
]


def test_token_f1_basic():
    assert token_f1("Nhớ mang tài liệu", "nhớ mang tài liệu") == 1.0
    assert token_f1("a b c d", "a b") == 2 * (0.5 * 1.0) / 1.5
    assert token_f1("", "") == 1.0
    assert token_f1("x", "y") == 0.0


def test_token_f1_nfc():
    decomposed = "Hường"  # "Hường" written with combining marks
    assert token_f1(decomposed, "hường") == 1.0


def test_exact_copy_scores_one():
    s = score_sample(GOLD, H, GOLD, H, FUZZY)
    assert (s.score, s.matched, s.unmatched) == (1.0, 3, 0)


def test_extra_read_costs_precision():
    pred = GOLD + [{"name": "contacts.search", "arguments": {"query": "Minh"}}]
    assert round(score_sample(pred, H, GOLD, H, FUZZY).score, 3) == 0.857


def test_missing_read_costs_recall():
    assert round(score_sample(GOLD[1:], H, GOLD, H, FUZZY).score, 3) == 0.8


def test_hash_mismatch_is_zero():
    assert score_sample(GOLD, "x" * 64, GOLD, H, FUZZY).score == 0.0


def test_empty_gold():
    assert score_sample([], "any", [], H, FUZZY).score == 1.0
    assert score_sample(GOLD[:1], H, [], H, FUZZY).score == 0.0


def test_no_match_is_zero_even_with_hash():
    pred = [{"name": "smart_home.get_status", "arguments": {"room": "phòng khách"}}]
    assert score_sample(pred, H, GOLD[:1], H, FUZZY).score == 0.0


def test_fuzzy_threshold():
    g = {"name": "sms.send", "arguments": {"contact_id": "c_10", "content": "nhớ mang tài liệu nhé"}}
    ok = {"name": "sms.send", "arguments": {"contact_id": "c_10", "content": "nhớ mang tài liệu"}}  # F1 = 0.889
    bad = {"name": "sms.send", "arguments": {"contact_id": "c_10", "content": "mang tài"}}  # F1 = 0.571
    assert calls_match(ok, g, FUZZY)
    assert not calls_match(bad, g, FUZZY)


def test_non_fuzzy_must_be_identical():
    g = {"name": "contacts.search", "arguments": {"query": "Minh"}}
    assert not calls_match({"name": "contacts.search", "arguments": {"query": "minh"}}, g, FUZZY)
    assert not calls_match({"name": "contacts.search", "arguments": {}}, g, FUZZY)


def test_empty_values_count_as_absent():
    g = {"name": "calendar.list_events", "arguments": {}}
    assert calls_match({"name": "calendar.list_events", "arguments": {"date_from": None}}, g, FUZZY)


def test_each_call_matches_once_and_order_ignored():
    read = {"name": "fitness.get_activity", "arguments": {"date": "2026-11-10"}}
    gold = [read, read, {"name": "fitness.set_goal", "arguments": {"steps": 9000}}]
    pred = [gold[2], read, read, read]
    assert max_matching(pred, gold, FUZZY) == 3
    s = score_sample(pred, H, gold, H, FUZZY)
    assert (s.matched, s.unmatched) == (3, 1)


def test_matching_is_maximum_not_greedy():
    # pred[0] fits both gold calls, pred[1] only the first: greedy pred[0]->gold[0] would leave pred[1] unmatched
    gold = [{"name": "sms.send", "arguments": {"contact_id": "c1", "content": "a b c"}},
            {"name": "sms.send", "arguments": {"contact_id": "c1", "content": "a b c d"}}]
    pred = [{"name": "sms.send", "arguments": {"contact_id": "c1", "content": "a b c d"}},
            {"name": "sms.send", "arguments": {"contact_id": "c1", "content": "a b"}}]
    assert max_matching(pred, gold, FUZZY) == 2
