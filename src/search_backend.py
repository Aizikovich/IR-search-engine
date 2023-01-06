import numpy as np
import pandas as pd
import bz2
from functools import partial
from collections import Counter, OrderedDict
import pickle
import heapq
from itertools import islice, count, groupby
from xml.etree import ElementTree
import codecs
import csv
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from time import time
import hashlib

import os
from pathlib import Path


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


nltk.download('stopwords')

"""
(a) cosine similarity using tf-idf on the body of articles,
(b) binary ranking using the title of articles,
(c) binary ranking using the anchor text,
(d) ranking by PageRank,
(e) ranking by article page views
"""

pkl_file = "../data/part15_preprocessed.pkl"

try:
    if os.environ["assignment_2_data"] is not None:
        pkl_file = Path(os.environ["assignment_2_data"])
except:
    Exception("Problem with one of the variables")

with open(pkl_file, 'rb') as f:
    pages = pickle.load(f)

print(pages[0][-1])

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower())]


def count_words(pages):
    """ Count words in the text of articles' title, body, and anchor text using
        the above `tokenize` function.
    Parameters:
    -----------
    pages: list of tuples
      Each tuple is a wiki article with id, title, body, and
      [(target_article_id, anchor_text), ...].
    Returns:
    --------
    list of str
      A list of tokens
    """
    word_counts = Counter()
    for wiki_id, title, body, links in pages:
        tokens = tokenize(title)
        token_body = tokenize(body)
        word_counts += Counter(tokens) + Counter(token_body)
        # tokenize body and anchor text and count

    return word_counts

# A function that builds an inverted index from a list of pages.
# reads N bytes at a time, and returns a generator of tokens
def read_in_chunks(file_object, chunk_size=1024):
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def inverted_index(pages):
    """ Build an inverted index from a list of pages.
    Parameters:
    -----------
    pages: list of tuples
      Each tuple is a wiki article with id, title, body, and
      [(target_article_id, anchor_text), ...].
    Returns:
    --------
    dict
      A dictionary of tokens to a list of tuples (wiki_id, count).
    """
    index = {}
    for wiki_id, title, body, links in pages:
        tokens = tokenize(title)
        token_body = tokenize(body)
        for token in tokens:
            if token in index:
                index[token].append((wiki_id, 1))
            else:
                index[token] = [(wiki_id, 1)]
        for token in token_body:
            if token in index:
                index[token].append((wiki_id, 1))
            else:
                index[token] = [(wiki_id, 1)]
    return index



# cosine similarity using tf-idf on the body of articles
def tfidf(pages, index, word_counts):
    """ Compute tf-idf for each article in the pages.
    Parameters:
    -----------
    pages: list of tuples
      Each tuple is a wiki article with id, title, body, and
      [(target_article_id, anchor_text), ...].
    index: dict
      A dictionary of tokens to a list of tuples (wiki_id, count).
    word_counts: dict
      A dictionary of tokens to their counts.
    Returns:
    --------
    dict
      A dictionary of wiki_id to a list of tuples (token, tf-idf score).
    """
    tfidf_dict = {}
    for wiki_id, title, body, links in pages:
        tokens = tokenize(title)
        token_body = tokenize(body)
        tfidf_dict[wiki_id] = []
        for token in tokens:
            tfidf_dict[wiki_id].append((token, (1 + np.log10(word_counts[token])) * np.log10(len(pages) / len(index[token]))))
        for token in token_body:
            tfidf_dict[wiki_id].append((token, (1 + np.log10(word_counts[token])) * np.log10(len(pages) / len(index[token]))))
    return tfidf_dict

