import pickle

from collections import Counter
from pathlib import Path

import pandas as pd

# Paths

# TODO put real path here + download wiki page views data like we did in assignment 1

yuval = 'C:/Users/Yuval/Documents/IR-finalP/data/pageviews-202108-user.pkl'
eran = 'C:/Users/Eran Aizikovich/Desktop/Courses/IR/final_proj/data/pageviews-202108-user.pkl'

basic_dir = eran
PV_PATH = Path(basic_dir)
wid2pv = Counter()

print("Loading page views...")
with open(PV_PATH, 'rb') as f:
    wid2pv = pickle.load(f)
# Todo: is this the fastest way to do this?
print("Page views loaded successfully!")


def page_views(ids, with_id=False):
    """
    Returns the number of page views that each of the provide wiki articles
    had in August 2021.

    Args:
    -----
        ids: list of ints
            list of wiki article IDs

    Returns:
    --------
        list of ints:
            list of page view numbers from August 2021 that correspond to the
            provided list article IDs.
    """
    # TODO test if this works as expected
    res = []
    for pid in ids:
        try:
            res.append((pid, wid2pv[pid])) if with_id else res.append(wid2pv[pid])
        except KeyError:
            res.append((pid, None)) if with_id else res.append(None)
    return res

    
