"""Text normalization for informal product reviews.

Stages (each can be switched on/off so the evaluation can ablate them):

  clean        Always on. Decodes literal unicode escapes (e.g. "\\u2019") and
               collapses whitespace. Not slang-related; applied to every config so
               ablations isolate the effect of slang handling.
  slang        Whole-token replacement from a slang lexicon using ONE pre-compiled
               regex (longest key first). Tokens are delimited with look-arounds
               rather than \\b, so keys that start/end with punctuation ("$",
               "c/o", "tl;dr") match correctly - the v1 \\b version silently
               skipped them.
  elongation   "goooood" -> "good", "sooooo" -> "soo": runs of 3+ identical letters
               are cut to 2, then to 1 if only the 1-letter form is a known word.
  emoji        Emoji -> text description ("😂" -> "face with tears of joy"), so
               both the lexicon and the classifier can read them.

Case is preserved (replacements are case-insensitive), because VADER uses
ALL-CAPS as an intensity cue; the v1 pipeline lowercased first and lost it.
"""
import json
import os
import re
from functools import lru_cache

import demoji

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
LEGACY_SLANG = os.path.join(DATA_DIR, "slangWords.json")
EXTENSION = os.path.join(DATA_DIR, "slang_ecommerce_extension.json")

_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_ELONGATED = re.compile(r"([a-zA-Z])\1{2,}")


def load_lexicon(version="v2"):
    """v1 = the original list exactly as written; v2 = v1 + e-commerce extension."""
    with open(LEGACY_SLANG, encoding="utf-8") as f:
        lex = {k.lower(): v.strip() for k, v in json.load(f).items()}
    if version == "v1":
        return lex
    with open(EXTENSION, encoding="utf-8") as f:
        ext = json.load(f)
    for key in ext["remove"]:
        lex.pop(key, None)
    for section in ("override", "hinglish", "misspellings", "informal"):
        lex.update({k.lower(): v for k, v in ext[section].items()})
    return lex


@lru_cache(maxsize=1)
def _known_words():
    words = set()
    try:
        from nltk.corpus import words as nltk_words
        words |= {w.lower() for w in nltk_words.words()}
    except LookupError:
        pass
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        words |= set(SentimentIntensityAnalyzer().lexicon)
    except LookupError:
        pass
    return words


class Normalizer:
    def __init__(self, slang=None, elongation=False, emoji=False):
        """slang: None, "v1" (original list) or "v2" (v1 list + extension)."""
        self.slang_version = slang
        self.elongation = elongation
        self.emoji = emoji
        self.lexicon = load_lexicon(slang) if slang else {}
        if self.lexicon:
            keys = sorted(self.lexicon, key=len, reverse=True)
            self._pattern = re.compile(
                r"(?<![\w])(" + "|".join(map(re.escape, keys)) + r")(?![\w])", re.IGNORECASE)

    @property
    def name(self):
        parts = [f"slang-{self.slang_version}" if self.slang_version else "no-slang"]
        if self.elongation:
            parts.append("elong")
        if self.emoji:
            parts.append("emoji")
        return "+".join(parts)

    @staticmethod
    def clean(text):
        text = _ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), str(text))
        return re.sub(r"\s+", " ", text).strip()

    def slang_hits(self, text):
        """Slang tokens found in the text (used for the slang-subset analysis)."""
        if not self.lexicon:
            return []
        return [m.group(1).lower() for m in self._pattern.finditer(self.clean(text))]

    def _fix_elongation(self, text):
        known = _known_words()

        def fix(word):
            two = _ELONGATED.sub(r"\1\1", word)
            if two.lower() in known or not known:
                return two
            one = _ELONGATED.sub(r"\1", word)
            return one if one.lower() in known else two

        return re.sub(r"[A-Za-z]+", lambda m: fix(m.group(0)) if _ELONGATED.search(m.group(0)) else m.group(0), text)

    def _replace(self, match):
        word = match.group(1)
        meaning = self.lexicon[word.lower()]
        # keep shouting as shouting: "AWSM" -> "AWESOME" (VADER boosts ALL-CAPS words)
        return meaning.upper() if len(word) > 1 and word.isupper() and word.isalpha() else meaning

    def __call__(self, text):
        text = self.clean(text)
        if self.elongation:
            text = self._fix_elongation(text)
        if self.lexicon:
            text = self._pattern.sub(self._replace, text)
        if self.emoji:
            text = demoji.replace_with_desc(text, sep=" ")
        return re.sub(r"\s+", " ", text).strip()
