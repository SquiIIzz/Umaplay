"""Tests for the StyleScheduleManager class."""

import pytest
from core.utils.style_schedule import StyleScheduleManager, StyleScheduleEntry
from core.utils.date_uma import DateInfo


def make_date(year_code: int, month: int, half: int) -> DateInfo:
    """Helper to create DateInfo for tests."""
    return DateInfo(
        raw=f"Test:Y{year_code}-{month}-{half}",
        year_code=year_code,
        month=month,
        half=half,
    )


class TestStyleScheduleEntry:
    def test_from_dict(self):
        entry = StyleScheduleEntry.from_dict({
            "yearCode": 2,
            "month": 6,
            "half": 1,
            "style": "pace",
        })
        assert entry.year_code == 2
        assert entry.month == 6
        assert entry.half == 1
        assert entry.style == "pace"

    def test_as_date_info(self):
        entry = StyleScheduleEntry(year_code=2, month=6, half=1, style="pace")
        di = entry.as_date_info()
        assert di.year_code == 2
        assert di.month == 6
        assert di.half == 1


class TestStyleScheduleManager:
    def test_empty_schedule_returns_debut_style(self):
        mgr = StyleScheduleManager(schedule=[], debut_style="front")
        date = make_date(year_code=2, month=6, half=1)
        assert mgr.get_style_for_date(date) == "front"

    def test_empty_schedule_no_debut_returns_none(self):
        mgr = StyleScheduleManager(schedule=[], debut_style=None)
        date = make_date(year_code=2, month=6, half=1)
        assert mgr.get_style_for_date(date) is None

    def test_schedule_overrides_at_correct_date(self):
        mgr = StyleScheduleManager(
            schedule=[
                {"yearCode": 2, "month": 6, "half": 1, "style": "late"},
            ],
            debut_style="front",
        )

        # Before schedule date
        before = make_date(year_code=2, month=5, half=2)
        assert mgr.get_style_for_date(before) == "front"

        # At schedule date
        at = make_date(year_code=2, month=6, half=1)
        assert mgr.get_style_for_date(at) == "late"

        # After schedule date
        after = make_date(year_code=3, month=1, half=1)
        assert mgr.get_style_for_date(after) == "late"

    def test_multiple_schedule_entries(self):
        mgr = StyleScheduleManager(
            schedule=[
                {"yearCode": 2, "month": 6, "half": 1, "style": "pace"},
                {"yearCode": 3, "month": 1, "half": 1, "style": "end"},
            ],
            debut_style="front",
        )

        # Junior year - uses debut style
        junior = make_date(year_code=1, month=12, half=2)
        assert mgr.get_style_for_date(junior) == "front"

        # Classic late - uses first schedule entry
        classic_late = make_date(year_code=2, month=9, half=1)
        assert mgr.get_style_for_date(classic_late) == "pace"

        # Senior - uses second schedule entry
        senior = make_date(year_code=3, month=6, half=1)
        assert mgr.get_style_for_date(senior) == "end"

    def test_should_apply_tracks_last_applied(self):
        mgr = StyleScheduleManager(
            schedule=[{"yearCode": 2, "month": 6, "half": 1, "style": "late"}],
            debut_style="front",
        )

        date = make_date(year_code=2, month=6, half=1)

        # First check should want to apply
        should, style = mgr.should_apply_style(date)
        assert should is True
        assert style == "late"

        # Mark as applied
        mgr.mark_applied("late")

        # Second check should not want to apply (already applied)
        should, style = mgr.should_apply_style(date)
        assert should is False
        assert style is None

    def test_should_apply_changes_when_style_changes(self):
        mgr = StyleScheduleManager(
            schedule=[
                {"yearCode": 2, "month": 6, "half": 1, "style": "pace"},
                {"yearCode": 3, "month": 1, "half": 1, "style": "end"},
            ],
            debut_style="front",
        )

        # Apply at classic
        date1 = make_date(year_code=2, month=6, half=1)
        should, style = mgr.should_apply_style(date1)
        assert should is True
        assert style == "pace"
        mgr.mark_applied("pace")

        # Same style, shouldn't apply
        date2 = make_date(year_code=2, month=9, half=1)
        should, style = mgr.should_apply_style(date2)
        assert should is False

        # New style at senior, should apply
        date3 = make_date(year_code=3, month=1, half=1)
        should, style = mgr.should_apply_style(date3)
        assert should is True
        assert style == "end"

    def test_get_debut_style(self):
        mgr = StyleScheduleManager(schedule=[], debut_style="front")
        assert mgr.get_debut_style() == "front"

        mgr2 = StyleScheduleManager(schedule=[], debut_style=None)
        assert mgr2.get_debut_style() is None

    def test_reset(self):
        mgr = StyleScheduleManager(
            schedule=[{"yearCode": 2, "month": 6, "half": 1, "style": "late"}],
            debut_style="front",
        )

        date = make_date(year_code=2, month=6, half=1)

        # Apply and mark
        mgr.should_apply_style(date)
        mgr.mark_applied("late")

        # Should not apply again
        should, _ = mgr.should_apply_style(date)
        assert should is False

        # Reset
        mgr.reset()

        # Should apply again after reset
        should, style = mgr.should_apply_style(date)
        assert should is True
        assert style == "late"

    def test_has_schedule(self):
        mgr1 = StyleScheduleManager(schedule=[], debut_style="front")
        assert mgr1.has_schedule() is False

        mgr2 = StyleScheduleManager(
            schedule=[{"yearCode": 2, "month": 6, "half": 1, "style": "pace"}],
            debut_style="front",
        )
        assert mgr2.has_schedule() is True

    def test_invalid_style_in_schedule_skipped(self):
        mgr = StyleScheduleManager(
            schedule=[
                {"yearCode": 2, "month": 6, "half": 1, "style": "invalid"},
                {"yearCode": 3, "month": 1, "half": 1, "style": "end"},
            ],
            debut_style="front",
        )
        # Invalid style is skipped, only one entry remains
        assert len(mgr.entries) == 1
        assert mgr.entries[0].style == "end"

    def test_invalid_debut_style_becomes_none(self):
        mgr = StyleScheduleManager(schedule=[], debut_style="invalid")
        assert mgr.debut_style is None

    def test_entries_sorted_by_date(self):
        mgr = StyleScheduleManager(
            schedule=[
                {"yearCode": 3, "month": 1, "half": 1, "style": "end"},
                {"yearCode": 2, "month": 6, "half": 1, "style": "pace"},
                {"yearCode": 1, "month": 12, "half": 2, "style": "late"},
            ],
            debut_style="front",
        )
        # Entries should be sorted by date
        assert mgr.entries[0].year_code == 1
        assert mgr.entries[1].year_code == 2
        assert mgr.entries[2].year_code == 3
