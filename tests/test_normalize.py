import pytest

from lifemate.normalize import normalize_text, resolve_dates

NOW = "2026-11-16T11:30:00+07:00"  # a Monday


@pytest.mark.parametrize("src,want", [
    ("chín trăm nghìn", "900000"), ("hai triệu rưỡi", "2500000"), ("131 triệu", "131000000"), ("50k", "50000"),
    ("hai chục", "20"), ("tháng chín", "tháng 9"), ("hai mươi mốt tuổi", "21 tuổi"),
    ("bảy giờ tối nay", "19:00 tối nay"), ("sáu rưỡi sáng mai", "06:30 sáng mai"), ("chín giờ kém mười lăm", "08:45"),
    ("mười một giờ đêm", "23:00"), ("sáng mốt tám giờ", "sáng mốt 08:00"),
    ("nhắc anh Mười", "nhắc anh Mười"), ("không đủ, một cái", "không đủ, một cái"),
])
def test_normalize_text(src, want):
    assert normalize_text(src) == want


@pytest.mark.parametrize("src,want", [
    ("mai", "2026-11-17"), ("sáng mốt", "2026-11-18"), ("thứ sáu", "2026-11-20"),
    ("thứ năm tuần sau", "2026-11-26"), ("ngày 20 tháng 12", "2026-12-20"), ("hôm mười bảy", "2026-11-17"),
])
def test_resolve_dates(src, want):
    assert resolve_dates(normalize_text(src), NOW)[0]["date"] == want


def test_name_mai_is_not_tomorrow():
    assert resolve_dates("Cô giáo Mai", NOW) == []


@pytest.mark.parametrize("src,want", [("hai triệu tám", "2800000"), ("một triệu hai", "1200000"),
                                       ("hai triệu hai trăm", "2200000"), ("tám giờ tối nay", "20:00 tối nay")])
def test_colloquial_money_and_kept_day_word(src, want):
    assert normalize_text(src) == want


def test_sentence_initial_mai_is_tomorrow():
    assert resolve_dates("Mai mưa thì huỷ", NOW)[0]["date"] == "2026-11-17"
