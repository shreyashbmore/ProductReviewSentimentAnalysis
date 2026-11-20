"""Command-line demo:  python scripts/predict.py "phone is awsm but bettery is wrost"  """
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.functions.sentiment import analyze  # noqa: E402

if __name__ == "__main__":
    for text in sys.argv[1:] or ["nyc phn, paisa wasool!!", "bakwas product, bettery drains in 2 hrs"]:
        print(json.dumps({"text": text, **analyze(text)}, ensure_ascii=False, indent=2))
