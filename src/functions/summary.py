"""Rule-based buying recommendation from a product's classified reviews.

Replaces the v1 version, which sent all reviews to OpenAI's text-davinci-003
(retired January 2024). This version is deterministic and explainable: it uses the
sentiment distribution plus the most frequent content words in positive vs
negative reviews.
"""
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STOP = ENGLISH_STOP_WORDS | {"product", "phone", "mobile", "good", "bad", "great", "nice", "best",
                              "worst", "really", "just", "like", "buy", "bought", "use", "using", "item"}


def distribution(reviews):
    counts = Counter(r.get("sentiment") for r in reviews)
    total = sum(counts[k] for k in ("positive", "negative", "neutral"))
    pct = {k: round(100 * counts[k] / total, 1) if total else 0.0 for k in ("positive", "negative", "neutral")}
    return {k: counts[k] for k in ("positive", "negative", "neutral")}, pct, total


def _themes(reviews, label, k=3):
    words = Counter()
    for r in reviews:
        if r.get("sentiment") == label:
            words.update({w for w in re.findall(r"[a-z]{3,}", r.get("review", "").lower()) if w not in _STOP})
    return [w for w, _ in words.most_common(k)]


def review_summary(reviews):
    _, pct, total = distribution(reviews)
    if total == 0:
        return "No reviews yet for this product."
    if pct["positive"] >= 60 and pct["negative"] <= 20:
        verdict = "Recommended"
    elif pct["negative"] >= 40:
        verdict = "Not recommended"
    else:
        verdict = "Mixed opinions: check the details that matter to you"
    text = (f"{verdict}. Based on {total} reviews: {pct['positive']}% positive, "
            f"{pct['negative']}% negative, {pct['neutral']}% neutral.")
    liked, disliked = _themes(reviews, "positive"), _themes(reviews, "negative")
    if liked:
        text += f" Buyers praise: {', '.join(liked)}."
    if disliked:
        text += f" Common complaints: {', '.join(disliked)}."
    return text
