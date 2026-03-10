"""
Style schedule management for dynamic running style changes during training runs.

Allows users to configure style changes at specific dates (year/month/half).
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from core.utils.date_uma import DateInfo, date_cmp
from core.utils.logger import logger_uma


@dataclass
class StyleScheduleEntry:
    """A scheduled style change at a specific date."""

    year_code: int  # 0=Pre-debut, 1=Junior, 2=Classic, 3=Senior, 4=Final
    month: int  # 1-12
    half: int  # 1=Early, 2=Late
    style: str  # 'end' | 'late' | 'pace' | 'front'

    def as_date_info(self) -> DateInfo:
        """Convert to DateInfo for comparison."""
        return DateInfo(
            raw=f"Schedule:Y{self.year_code}-{self.month}-{self.half}",
            year_code=self.year_code,
            month=self.month,
            half=self.half,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StyleScheduleEntry":
        """Create from config dict."""
        return cls(
            year_code=int(d.get("yearCode", 1)),
            month=int(d.get("month", 1)),
            half=int(d.get("half", 1)),
            style=str(d.get("style", "pace")),
        )


class StyleScheduleManager:
    """
    Manages date-based running style changes.

    Tracks the current active style based on a schedule and ensures
    style changes are only applied when the style actually changes.
    """

    VALID_STYLES = ("end", "late", "pace", "front")

    def __init__(
        self,
        schedule: Optional[List[Dict[str, Any]]] = None,
        debut_style: Optional[str] = None,
    ):
        """
        Initialize the style schedule manager.

        Args:
            schedule: List of dicts with yearCode, month, half, style keys
            debut_style: The style to use for debut race (and as initial default)
        """
        self.debut_style = debut_style if debut_style in self.VALID_STYLES else None

        # Parse and sort entries by date
        self.entries: List[StyleScheduleEntry] = []
        if schedule:
            for entry_dict in schedule:
                try:
                    entry = StyleScheduleEntry.from_dict(entry_dict)
                    if entry.style in self.VALID_STYLES:
                        self.entries.append(entry)
                    else:
                        logger_uma.warning(
                            f"[StyleSchedule] Invalid style '{entry.style}', skipping"
                        )
                except (KeyError, ValueError, TypeError) as e:
                    logger_uma.warning(f"[StyleSchedule] Invalid entry {entry_dict}: {e}")

        # Sort by date (earliest first)
        self.entries.sort(key=lambda e: (e.year_code, e.month, e.half))

        # Track what was last applied to avoid redundant clicks
        self._last_applied_style: Optional[str] = None

        logger_uma.debug(
            f"[StyleSchedule] Initialized with debut_style={debut_style}, "
            f"{len(self.entries)} scheduled changes"
        )

    def get_style_for_date(self, date: DateInfo) -> Optional[str]:
        """
        Get the style that should be active at the given date.

        Finds the latest schedule entry that is <= the current date.
        Falls back to debut_style if no schedule entry matches.

        Args:
            date: Current career date

        Returns:
            The style to use, or None if no style is configured
        """
        active_style = self.debut_style

        for entry in self.entries:
            entry_date = entry.as_date_info()
            cmp_result = date_cmp(entry_date, date)

            if cmp_result <= 0:
                # Entry date is before or equal to current date
                active_style = entry.style
            else:
                # Entry is in the future, stop checking
                # (entries are sorted, so all remaining are also future)
                break

        return active_style

    def should_apply_style(
        self, date: DateInfo
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if a style should be applied at the current date.

        Only returns True if the style differs from the last applied style.

        Args:
            date: Current career date

        Returns:
            Tuple of (should_apply, style_to_apply)
        """
        style = self.get_style_for_date(date)

        if style and style != self._last_applied_style:
            logger_uma.debug(
                f"[StyleSchedule] Style change needed at {date.as_key()}: "
                f"{self._last_applied_style} -> {style}"
            )
            return True, style

        return False, None

    def get_debut_style(self) -> Optional[str]:
        """Get the style for debut race."""
        return self.debut_style

    def mark_applied(self, style: str) -> None:
        """
        Mark a style as having been applied.

        Call this after successfully applying a style to prevent
        redundant style changes.

        Args:
            style: The style that was applied
        """
        logger_uma.debug(f"[StyleSchedule] Marked style as applied: {style}")
        self._last_applied_style = style

    def reset(self) -> None:
        """
        Reset tracking state.

        Call this at the start of a new career run.
        """
        logger_uma.debug("[StyleSchedule] Reset tracking state")
        self._last_applied_style = None

    def has_schedule(self) -> bool:
        """Return True if there are any scheduled style changes."""
        return len(self.entries) > 0

    def __repr__(self) -> str:
        return (
            f"StyleScheduleManager(debut={self.debut_style}, "
            f"entries={len(self.entries)}, last_applied={self._last_applied_style})"
        )
