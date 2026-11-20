"""Sentiment inference used by the web app.

Loads the trained model (models/sentiment_model.joblib, produced by
scripts/evaluate.py): slang/elongation/emoji normalization -> word + char TF-IDF
+ VADER features -> multinomial logistic regression.
"""
import os
from functools import lru_cache

import joblib

MODEL_PATH = os.environ.get(
    "SENTIMENT_MODEL", os.path.join(os.path.dirname(__file__), "..", "..", "models", "sentiment_model.joblib"))


@lru_cache(maxsize=1)
def _model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run: python scripts/download_data.py && python scripts/evaluate.py")
    return joblib.load(MODEL_PATH)


def analyze(text):
    """Return label, confidence, class probabilities and the normalized text."""
    model = _model()
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    best = int(proba.argmax())
    return {
        "label": classes[best],
        "confidence": round(float(proba[best]), 3),
        "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
        "normalized": model.named_steps["normalize"].transform([text])[0],
    }


def detect_sentiment(text):
    return analyze(text)["label"]
