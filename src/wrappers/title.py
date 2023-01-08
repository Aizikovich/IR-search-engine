from pathlib import Path

from src.methods.binary_search_methods import binary_search
from src.invertedIndex import InvertedIndex
# from data.create_index import InvertedIndex

# import sys
# print(sys.path)

# TODO load index here and base dir
base_dir = Path('C:/Users/Yuval/Documents/IR-finalP/data/title_index')
name = 'wiki_index'
print("Loading title index...")
titleIndex = InvertedIndex.read_index(base_dir, name)
words, pls = zip(*titleIndex.posting_lists_iter(base_dir))
print("Title index loaded successfully!")
print(words)
print(pls)

def search_title_by_query(query: str, n=0) -> list:
    """
    binary search for titles by query
    Parameters:
    ----------
    query: str
        query to search example: 'Can you give us more GCP credits?'
    n: int
        number of top documents to return if n=0 return all documents
    Returns:
    --------
    list of top n doc_id
    """

    return binary_search(query, words, pls, n)


