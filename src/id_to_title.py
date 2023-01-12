from pathlib import Path

import pandas as pd

yuval = 'C:/Users/Yuval/Documents/IR-finalP/data/title_names.csv'
eran = 'C:/Users/Eran Aizikovich/Desktop/Courses/IR/final_proj/data/title_names.csv'
basic_dir = eran
base_dir = Path(basic_dir)
print("Loading title to id file...")
title_to_id = pd.read_csv(base_dir, header=None, names=['id', 'title'])
titles_dict = dict(zip(title_to_id['id'], title_to_id['title']))
print("Title to id file loaded successfully!")


def get_titles(doc_ids: list):
    """
    This function convert a list of doc_id to a list of (doc_id, title) tuples.
    Parameters:
    ----------
    doc_ids: list
        list of doc_id
    Returns:
    --------
    list of titles corresponding to the doc_ids
    """
    return [(doc_id, titles_dict.get(int(doc_id), None)) for doc_id in doc_ids]

