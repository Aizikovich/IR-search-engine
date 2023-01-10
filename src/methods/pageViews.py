import pickle

from collections import Counter
from pathlib import Path

import pandas as pd

# Paths

# TODO put real path here + download wiki page views data like we did in assignment 1
basic_dir = 'C:/Users/Yuval/Documents/'

PV_PATH = Path(basic_dir + 'IR-finalP/data/pageviews-202108-user.pkl')
wid2pv = Counter()

print("Loading page views...")
with open(PV_PATH, 'rb') as f:
    wid2pv = pickle.load(f)
# Todo: is this the fastest way to do this?
print("Page views loaded successfully!")


def page_views(ids):
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
            res.append(wid2pv[pid])
        except KeyError:
            res.append(None)
    return res

    
