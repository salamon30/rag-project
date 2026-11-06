"""Usage caps for the public demo, so visitors cannot use up the API key.

Off unless DEMO_DAILY_LIMIT is set (the Hugging Face Space sets it); local runs are unlimited.
"""
import datetime
import json
from pathlib import Path


class DailyCounter:
    """Counts questions per UTC day in a small JSON file and refuses once the limit is reached."""

    def __init__(self, path: Path, limit: int):
        self.path = Path(path)
        self.limit = limit

    def _used(self, today: str) -> int:
        try:
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            return 0
        return data["count"] if data.get("date") == today else 0

    def remaining(self, today: str | None = None) -> int:
        today = today or datetime.date.today().isoformat()
        return max(0, self.limit - self._used(today))

    def take(self, today: str | None = None) -> bool:
        """Use one question if any are left today. Returns False when the limit is reached."""
        today = today or datetime.date.today().isoformat()
        used = self._used(today)
        if used >= self.limit:
            return False
        self.path.write_text(json.dumps({"date": today, "count": used + 1}))
        return True
