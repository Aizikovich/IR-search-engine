import numpy as np
from src.methods.tokenizer import tokenize


def get_candidate_documents_for_binary(query_to_search, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through
    every token in query_to_search and calculate how many times it appears in each document.

    :param query_to_search: list of tokens (str).  Example: 'Hello, I love information retrival' --->
                                                    ['hello','love','information', 'retrieval']
    :param index:        inverted index loaded from the corresponding files.
    :param words:        list of words in the index
    :param pls:          posting list for every word in words
    :return:         dictionary of candidate documents and their scores


    'Hello, I love information retrival'
    doc1 = 'hello world'
    doc2 = 'hello world love '
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            for doc in list_of_doc:
                candidates[doc] = candidates.get(doc, 0) + 1

    return candidates


def get_top_n(candidates, n=0):
    """
    Get the top n documents from the candidates dictionary
    :param candidates: dictionary of candidate documents and their scores
    :param n: number of top documents to return
    :return: list of top n documents
    """
    sort_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    if n == 0:
        return sort_candidates
    return sort_candidates[:n]


def binary_search(query_to_search, index, words, pls, n=0):
    """
    Search for a query using binary search.

    :param n: number of top documents to return
    :param query_to_search: (str). Example: 'Can you give us more GCP credits?'
    :param index: inverted index loaded from the corresponding files.
    :param words: list of words in the index
    :param pls: posting list for every word in words
    :return: list of top n doc_id
    """
    query = tokenize(query_to_search)
    candidates = get_candidate_documents_for_binary(query, words, pls)
    return [x[0] for x in get_top_n(candidates, n)]


