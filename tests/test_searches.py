import unittest
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from src.methods.body_search import get_cosine_similarity
import json


def load_train_set():
    with open('queries_train.json') as f:
        train_set = json.load(f)
    return train_set


class TestStringMethods(unittest.TestCase):

    def test_cosine_sim(self):
        def cos_sim(a, b):
            return dot(a, b) / (norm(a) * norm(b))

        df = pd.DataFrame({f'c{i}': i for i in range(1, 6)}, index=[1, 2, 3, 4, 5])
        for _ in range(5):
            query = np.random.randint(1, 6, 5)
            round_res = round(get_cosine_similarity(query, df)[1][0][0], 5)
            self.assertEqual(round_res, round(cos_sim(query, df.loc[1].values), 5))

    def test_train_set(self):
        train_set = load_train_set()
        self.assertEqual(len(train_set), 30, msg='train set should contain 30 queries')
        self.assertTrue(isinstance(train_set, dict), msg='train_set is not a dictionary')

    def test_1(self):
        train = load_train_set()
        for k, v in train.items():
            print(k, v)


if __name__ == '__main__':
    unittest.main()
