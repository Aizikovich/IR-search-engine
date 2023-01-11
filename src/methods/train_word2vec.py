import pickle
from collections import defaultdict
from functools import reduce
from itertools import groupby
from src.id_to_title import get_titles

from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

from src.methods.tokenizer import tokenize

import gensim
import pandas as pd

yuval_path = 'C:/Users/Yuval/Documents/IR-finalP/data/'
eran_path = 'C:/Users/Eran Aizikovich/Desktop/Courses/IR/final_proj/data/'

name = 'word2Vec'
basic_dir = yuval_path





def main():
    # read the dump
    pkl_file = "part15_preprocessed.pkl"
    print("Creating Word2Vec model...")
    with open(yuval_path + pkl_file, 'rb') as f:
        pages = pickle.load(f)
        bodies = [page[2] for page in pages]
        bodies = list(map(tokenize, bodies))

        concStr = lambda x: reduce(lambda a, b: str(a) + " " + str(b), x, '')
        listToStr = list(map(concStr, bodies))
        # bodies to df
        ids = [pages[i][0] for i in range(len(pages))]

        df = pd.DataFrame(zip(ids, listToStr), columns=['id', 'text'])
        # apply gensim simple_preprocess
        df['text'] = df['text'].map(lambda x: gensim.utils.simple_preprocess(x))
        word2vec = gensim.models.Word2Vec(window=7,
                                          min_count=5,
                                          workers=4
                                          )
        word2vec.build_vocab(df['text'], progress_per=100)
        word2vec.train(df['text'], total_examples=word2vec.corpus_count, epochs=30, report_delay=1)
        word2vec.save(basic_dir + name + '.model')
        print("Word2Vec model created successfully!")
        print("creating Doc2Vec model...")
        doc2vec = gensim.models.KeyedVectors.load_word2vec_format(basic_dir + name + '.model', binary=True)
        doc2vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=40)

def main2():
    # read the dump
    pkl_file = "part15_preprocessed.pkl"
    print("reading corpus for Doc2Vec model...")
    with open(yuval_path + pkl_file, 'rb') as f:
        pages = pickle.load(f)
        bodies = [page[2] for page in pages]
        bodies = list(map(tokenize, bodies))

        # bodies to df
        ids = [pages[i][0] for i in range(len(pages))]

        df = pd.DataFrame(zip(ids, bodies), columns=['id', 'text'])
        # apply gensim simple_preprocess
        print("creating Doc2Vec model...")
        # Create a list of labeled documents
        labeled_documents = [TaggedDocument(row['text'], [row['id']]) for i, row in df.iterrows()]
        doc2vec_model = gensim.models.Doc2Vec(dm=0,
                                              dbow_words=1,
                                              vector_size=200,
                                              window=9,
                                              min_count=20,
                                              workers=4,
                                              alpha=0.025,
                                              min_alpha=0.025)
        # Build vocabulary of the doc2vec model
        doc2vec_model.build_vocab(labeled_documents)
        # Train the doc2vec model on the labeled documents
        doc2vec_model.train(labeled_documents, total_examples=doc2vec_model.corpus_count, epochs=100)
        doc2vec_model.save(basic_dir + 'doc2vec' + '.model')
        print("Doc2Vec model created successfully!")


if __name__ == '__main__':
    main2()
    pkl_file = "part15_preprocessed.pkl"
    with open(yuval_path + pkl_file, 'rb') as f:
        pages = pickle.load(f)
        ids = [pages[i][0] for i in range(len(pages))]
        # print(pages[0])
    # print([page[2] for page in pages if page[0] == 17338108])
    # load the model
    model = gensim.models.Doc2Vec.load(basic_dir + 'doc2vec' + '.model')

    # Create a list of words for the query
    query_words = ['best', 'song']

    # Generate vector representation for the query
    query_vec = model.infer_vector(query_words)

    # get best matches using BM25

    # Get the vector representations for the retrieved documents
    doc_v = [model.dv[doc_id] for doc_id in ids]
    print(len(query_vec))
    # Calculate the similarity between the query and the retrieved documents
    similarities = cosine_similarity(doc_v, query_vec.reshape(1, -1))
    # zip ids and simalrity
    zipped = zip(ids, similarities)
    # sort by similarity
    sorted_zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    # get the top 10
    top10 = sorted_zipped[:20]
    # get there title
    titles = get_titles([t[0] for t in top10])
    print(titles)
    print(top10)
