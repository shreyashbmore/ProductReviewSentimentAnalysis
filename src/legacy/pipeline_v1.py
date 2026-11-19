"""The v1 sentiment pipeline, reproduced for benchmarking.

Faithful to the original src/functions/preprocess.py + sentiment.py:
lowercase -> slang replacement with a \\b regex per dictionary entry (227 entries,
data/slangWords.json) -> emoji to text -> VADER compound score, thresholds +/-0.05.

Omitted: the original 'sarcasm_correction' step, which re-labelled every review
with OpenAI's text-davinci-003. That model was retired by OpenAI in January 2024,
so the step cannot be run or benchmarked today.
"""
import json
import os
import re

import demoji

from src.nlp.model import vader

_SLANG = os.path.join(os.path.dirname(__file__), "..", "..", "data", "slangWords.json")
with open(_SLANG, encoding="utf-8") as f:
    _DICT = json.load(f)
_PATTERNS = [(re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE), " " + v + " ") for k, v in _DICT.items()]


def preprocess_v1(text):
    text = str(text).lower()
    for pattern, meaning in _PATTERNS:
        text = pattern.sub(meaning, text)
    return demoji.replace_with_desc(text, sep=" ")


def compound_v1(text):
    return vader().polarity_scores(preprocess_v1(text))["compound"]
