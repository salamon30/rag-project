from app.limits import DailyCounter


def test_counter_allows_up_to_the_limit(tmp_path):
    counter = DailyCounter(tmp_path / "usage.json", limit=2)
    assert counter.take("2026-11-06")
    assert counter.take("2026-11-06")
    assert not counter.take("2026-11-06")
    assert counter.remaining("2026-11-06") == 0


def test_counter_resets_on_a_new_day(tmp_path):
    counter = DailyCounter(tmp_path / "usage.json", limit=1)
    assert counter.take("2026-11-06")
    assert not counter.take("2026-11-06")
    assert counter.take("2026-11-07")


def test_missing_or_broken_file_counts_as_unused(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("not json")
    assert DailyCounter(path, limit=3).remaining("2026-11-06") == 3
