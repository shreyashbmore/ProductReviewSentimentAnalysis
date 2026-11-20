"""Review storage.

STORAGE_BACKEND=aws   -> the project's AWS API Gateway endpoint (Lambda + DynamoDB),
                         URL taken from REVIEW_API_URL.
STORAGE_BACKEND=local -> a JSON file (data/reviews_local.json), so the demo runs
                         without cloud credentials. This is the default.
"""
import json
import os
import threading

from src.api.getReview import execute_get_api
from src.api.postReview import execute_api

BACKEND = os.environ.get("STORAGE_BACKEND", "local")
LOCAL_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "reviews_local.json")
_lock = threading.Lock()


def _read_local():
    try:
        with open(LOCAL_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_review(record):
    if BACKEND == "aws":
        return execute_api(record)
    with _lock:
        rows = _read_local()
        rows.append(record)
        with open(LOCAL_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


def get_reviews(product_id):
    if BACKEND == "aws":
        data = execute_get_api(product_id)
        return data if isinstance(data, list) else []
    return [r for r in _read_local() if r.get("product_id") == product_id]
