import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim_using_sklearn(queries, tfidf):
    """
    In this function you need to utilize the cosine_similarity function from sklearn.
    You need to compute the similarity between the queries and the given documents.
    This function will return a DataFrame in the following shape: (# of queries, # of documents).
    Each value in the DataFrame will represent the cosine_similarity between given query and document.

    Parameters:
    -----------
      queries: sparse matrix represent the queries after transformation of tfidfvectorizer.
      documents: sparse matrix represent the documents.

    Returns:
    --------
      DataFrame: This function will return a DataFrame in the following shape: (# of queries, # of documents).
      Each value in the DataFrame will represent the cosine_similarity between given query and document.
    """
    sim_mat = cosine_similarity(queries, tfidf)
    return pd.DataFrame(sim_mat)


def bm25_preprocess(data):
    """
    This function goes through the data and saves relevant information for the calculation of bm25.
    Specifically, in this function, we will create 3 objects that gather information regarding document length, term frequency and
    document frequency.
    Parameters
    -----------
    data: list of lists. Each inner list is a list of tokens.
    Example of data:
    [
        ['sky', 'blue', 'see', 'blue', 'sun'],
        ['sun', 'bright', 'yellow'],
        ['comes', 'blue', 'sun'],
        ['lucy', 'sky', 'diamonds', 'see', 'sun', 'sky'],
        ['sun', 'sun', 'blue', 'sun'],
        ['lucy', 'likes', 'blue', 'bright', 'diamonds']
    ]

    Returns:
    -----------
    three objects as follows:
                a) doc_len: list of integer. Each element represents the length of a document.
                b) tf: list of dictionaries. Each dictionary corresponds to a document as follows:
                                                                    key: term
                                                                    value: normalized term frequency (by the length of document)


                c) df: dictionary representing the document frequency as follows:
                                                                    key: term
                                                                    value: document frequency
    """

    def normalized_words_in_doc(doc):
        return {k: doc.count(k) / len(doc) for k in doc}

    terms = set([t for d in data for t in d])

    def is_in(term, doc):
        return 1 if term in doc else 0

    doc_len = [len(doc) for doc in data]

    tf = [normalized_words_in_doc(doc) for doc in data]
    df = {term: sum([is_in(term, doc) for doc in data]) for term in terms}

    return doc_len, tf, df


class BM25:
    """
    Best Match 25.

    Parameters to tune
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.
        The frequnecy is normilzied by the max term frequency for each document.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.
    """

    def __init__(self, doc_len, df, tf=None, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.tf_ = tf
        self.doc_len_ = doc_len
        self.df_ = df
        self.N_ = len(doc_len)
        self.avgdl_ = sum(doc_len) / len(doc_len)

    def calc_idf(self, query):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """

        def formula(N, nti):
            return np.log((N - nti + 0.5) / (nti + 0.5) + 1)

        d = {term: formula(self.N_, self.df_[term]) for term in query}
        return d

    def search(self, queries):
        """
        This function use the _score function to calculate the bm25 score for all queries provided.

        Parameters:
        -----------
        queries: list of lists. Each inner list is a list of tokens. For example:
                                                                                    [
                                                                                        ['look', 'blue', 'sky'],
                                                                                        ['likes', 'blue', 'sun'],
                                                                                        ['likes', 'diamonds']
                                                                                    ]

        Returns:
        -----------
        list of scores of bm25
        """
        scores = []
        for query in queries:
            scores.append([self._score(query, doc_id) for doc_id in range(self.N_)])
        return scores

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """

        # YOUR CODE HERE

        def formula(ftd, k1, b, size_D, avgdl):
            return (ftd * (k1 + 1)) / (ftd + k1 * (1 - b + b * (size_D / avgdl)))

        scoers = []
        for term in query:
            if term in self.tf_[doc_id]:
                IDFs = self.calc_idf([term])[term]
                ftd = self.tf_[doc_id][term]
                size_D = self.doc_len_[doc_id]
                calc = formula(ftd, self.k1, self.b, size_D, self.avgdl_)
                scoers.append(IDFs * calc)

        return (sum(scoers))

