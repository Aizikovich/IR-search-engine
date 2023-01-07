from src.methods.binary_search_methods import binary_search
from src.invertedIndex import InvertedIndex

# TODO load index here and base dir
index_title = InvertedIndex('title')
base_dir = ' '
words, pls = zip(*index_title.posting_lists_iter(base_dir))


def search_title_by_query(query: str, n=0) -> list:
    """
    binary search for titles by query
    :param query: str - query to search at original form (not tokenized) example: 'Can you give us more GCP credits?'
    :param n: int - number of top documents to return if n=0 return all
    :return: list of top n documents matching the query
    """
    return binary_search(query, index_title, words, pls, n)


