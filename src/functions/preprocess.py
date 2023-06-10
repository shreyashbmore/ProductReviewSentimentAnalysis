import json
import re
import demoji


def replace_slang_words(text):
    print("input to slang"+text)
    with open('data/slangWords.json', 'r') as file:
        slang_dict = json.load(file)

    for slang, meaning in slang_dict.items():
        text = re.sub(r"\b" + re.escape(slang) + r"\b", " " + meaning + " ", text, flags=re.IGNORECASE)
    print("output of slang"+text)
    return text


def convert_emoji_to_word(text):
    print("input to emoji"+text)
    demoji.download_codes()  # Downloads the emoji code mappings if not already downloaded
    converted_text = demoji.replace_with_desc(text)
    print("output of emoji"+text)
    return converted_text

    
def preprocess_text(text):
    # Add your preprocessing steps here
    processed_text = text.lower()
    processed_text = replace_slang_words(processed_text)
    processed_text = convert_emoji_to_word(processed_text)
    return processed_text