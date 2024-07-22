import re

from nltk.tokenize import sent_tokenize


def extract_sentences(article):
    article_cleaned = _cleaning(article)
    sentences = _tokenize_sentences(article_cleaned)
    return sentences


def _cleaning(article):
    article_cleaned = re.sub(r'\\n', r'\n', article)
    return article_cleaned


def _tokenize_sentences(article):
    sentences = sent_tokenize(article)
    return sentences
