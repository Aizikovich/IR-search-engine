import pickle
import pandas as pd

# Paths

# TODO put real path here + download wiki page views data like we did in assignment 1
PV_PATH = '/pageviews-202108-user.pkl'

print("Loading page views...")
with open(PV_PATH, 'rb') as f:
    wid2pv = pickle.load(f)

pv = pd.DataFrame.from_dict(wid2pv, orient='index', columns=['p_id', 'pageview'])
pv = pv.set_index('p_id')
pv_dict = pv.to_dict()['pageview']
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
            res.append(pv_dict[pid])
        except KeyError:
            res.append(None)
    return res

    
