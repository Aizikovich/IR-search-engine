import math
from collections import Counter
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

from src.methods.tokenizer import tokenize

corpus_size = 6345849


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros(total_vocab_size)
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log(corpus_size / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def get_candidate_documents_and_scores(query_to_search, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]

            # DL[str(doc_id)] = for each doc_id, the length of the document
            normlized_tfidf = [(doc_id, tfidf) for doc_id, tfidf in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.


    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(index.term_total)
    # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    candidates_scores = get_candidate_documents_and_scores(query_to_search, words, pls)
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def get_top_n(sim_dict, n=0):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    result = sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1][0], reverse=True)
    if n == 0:
        return result
    return result[:n]


def get_cosine_similarity(q_vector, docs_matrix):
    """
    Calculate the cosine similarity between the query and the documents.
    :param q_vector:
    :param docs_matrix: data frame - each row is a document, each column is a term, each cell is the tfidf score.
    :return: dictionary - {key = doc id : value = cosine similarity score (q_vector & doc_vector)}
    """
    sim_dict = {}

    for doc_id in docs_matrix.index:
        doc_vector = docs_matrix.loc[doc_id].values
        sim_dict[doc_id] = cosine_similarity(q_vector.reshape(1, -1), doc_vector.reshape(1, -1))
    return sim_dict


def search_body_wiki(query, index, words, pls, n=0):
    """
    Search for a given query in the body of the documents.

    Parameters:
    -----------
    query: a string of the query to search for.
    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    query_tokenize = tokenize(query)
    doc_tfidf = generate_document_tfidf_matrix(query_tokenize, index, words, pls)
    query_tfidf = generate_query_tfidf_vector(query_tokenize, index)
    sim_score_dict = get_cosine_similarity(query_tfidf, doc_tfidf)

    return get_top_n(sim_score_dict, n)
