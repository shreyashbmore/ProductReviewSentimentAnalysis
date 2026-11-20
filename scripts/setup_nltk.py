"""One-time download of the NLTK resources the project needs."""
import nltk

for resource in ("vader_lexicon", "words"):
    nltk.download(resource, quiet=True)
print("NLTK resources ready.")
