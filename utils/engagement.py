import json
from datetime import datetime


def compute_engagement_score(history_item: dict) -> float:
    """
    Compute engagement score based on watch behavior.

    Factors:
    - timesWatched: how many times watched (weight: 0.5)
    - completion: how much of the movie was watched (weight: 1.0)
    - recency: how recently was watched (weight: 0.3)
    """
    movie_meta = history_item.get("movie", {})

    try:
        meta = json.loads(movie_meta.get("metadata", "{}"))
        duration = meta.get("duration", 1)
    except Exception:
        duration = 1

    times_watched = history_item.get("timesWatched", 0)
    last_watch_sec = history_item.get("lastWatchSeconds", 0)

    completion = min(last_watch_sec / duration, 1.0) if duration > 0 else 0

    modified_str = history_item.get("modifiedDate", "")

    try:
        modified_date = datetime.strptime(modified_str, "%d/%m/%Y %H:%M:%S")
        days_ago = (datetime.now() - modified_date).days
        recency = 1 / (1 + days_ago / 30)
    except Exception:
        recency = 0.5

    score = (
        times_watched * 0.5 +
        completion * 1.0 +
        recency * 0.3
    )

    return max(score, 0.01)
