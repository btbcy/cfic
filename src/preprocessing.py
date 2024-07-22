import re

from nltk.tokenize import sent_tokenize


def extract_sentences(article):
    article_cleaned = clean_text(article)
    sentences = article_to_sentences(article_cleaned)
    return sentences


def clean_text(article):
    article_cleaned = re.sub(r'\\n', r'\n', article)
    return article_cleaned


def article_to_sentences(article):
    sentences = sent_tokenize(article)
    return sentences
