from datetime import datetime
from typing import Optional, Tuple

def extract_day_hour(dt: Optional[datetime]) -> Optional[Tuple[int, int]]:
    # Given a datetime, return (weekday, hour)
    if dt is None:
        return None
    return dt.weekday(), dt.hour