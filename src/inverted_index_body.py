from collections import Counter
from contextlib import closing

import numpy as np

from invertedIndex import InvertedIndex, MultiFileWriter, MultiFileReader, TUPLE_SIZE


class InvertedIndexBody(InvertedIndex):
    N = 6345849

    def __init__(self, docs):
        super().__init__(docs)
        self.docs = docs
        self.docs_len = {}
        self.term2remove = []
        # filter words
        self.filter_terms_by_frequency()
        # calculate document length
        self.calculate_doc_len()

    def filter_terms_by_frequency(self):
        """ Filter out terms that appear in less than 50 documents. """
        # TODO: implement check if this is really works
        temp = {}
        for term, locs in self.posting_locs.items():
            if self.df[term] >= 50:
                temp[term] = locs
            else:
                self.term2remove.append(term)
                self.df.pop(term)
                self.term_total.pop(term)
        self.posting_locs = temp

    def calculate_doc_len(self):
        """ Calculate the length of each document. """
        for doc_id, tokens in self.docs.items():
            temp = []
            for token in tokens:
                if token not in self.term2remove:
                    temp.append(token)
            self.docs_len[doc_id] = len(temp)

    def posting_lists_iter(self, directory):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tfidf:float), ...]) tuple.
        """
        # self.df[term] - stores document frequency per term
        # for example if the term "free" appears in 4 different documents, then self.df["free"] = 4
        # corpus size - number of documents: N = 6345849
        N, epsilon = 6345849, 0.0000001
        f = True
        with closing(MultiFileReader()) as reader:
            for term, locs in self.posting_locs.items():
                # read a certain number of bytes into variable b
                b = reader.read(locs, self.df[term] * TUPLE_SIZE)
                posting_list = []
                # convert the bytes read into `b` to a proper posting list.

                for i in range(self.df[term]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    idf = np.log10(N / (self.df[term] + epsilon))
                    normalized_tf = tf / self.docs_len[doc_id]
                    posting_list.append((doc_id, normalized_tf * idf))

                yield term, posting_list
