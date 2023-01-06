from methods.binary_search_methods import binary_search
from src.invertedIndex import InvertedIndex
from data.tokenizer import tokenize
# TODO load index here
index_anchor = InvertedIndex('title')


def search_anchor_by_query(query: str, n=0) -> list:
    """
    binary search for anchors by query
    :param query: str - query to search at original form (not tokenized) example: 'Can you give us more GCP credits?'
    :param n: int - number of top documents to return if n=0 return all
    :return: list of top n documents matching the query
    """
    token_query = tokenize(query)
    # TODO implement in InvertedIndex posting_lists_tokens method
    words, pls = index_anchor.posting_lists_tokens(token_query)
    return binary_search(query, index_anchor, words, pls, n)
