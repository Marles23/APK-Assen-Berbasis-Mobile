# youtube_service.py
import os
from typing import List, Dict
import requests

YOUTUBE_API_KEY = os.environ.get("YT_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("Environment variable YT_API_KEY belum diset.")

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

def get_comments_for_video(
    video_id: str,
    max_comments: int = 100,
) -> List[Dict]:
    """
    Ambil komentar dari YouTube untuk 1 video.
    Return list dict: {author, text}
    """

    comments: List[Dict] = []
    page_token = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "key": YOUTUBE_API_KEY,
            "maxResults": 300,            # max per request
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(YOUTUBE_API_URL, params=params, timeout=10)
        if resp.status_code != 200:
            print("YouTube API error:", resp.status_code, resp.text)
            break

        data = resp.json()
        items = data.get("items", [])

        for item in items:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            author = snippet.get("authorDisplayName", "Unknown")
            text = snippet.get("textDisplay", "")
            if text:
                comments.append({
                    "author": author,
                    "text": text,
                })
                if len(comments) >= max_comments:
                    break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments
