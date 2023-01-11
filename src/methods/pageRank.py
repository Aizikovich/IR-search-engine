from pathlib import Path

import pandas as pd

# Paths

# TODO put real path here + download wiki page rank data like we did in assignment 3

yuval = 'C:/Users/Yuval/Documents/IR-finalP/data/pr_wiki.csv'
eran = 'C:/Users/Eran Aizikovich/Desktop/Courses/IR/final_proj/data/pr_wiki.csv'

basic_dir = yuval

PR_PATH = Path(basic_dir)
print("Loading page rank...")
pr = pd.read_csv(PR_PATH)
pr.columns = ['p_id', 'page_rank']
pr.set_index('p_id', inplace=True)
pr_dict = pr.to_dict()['page_rank']
print("Page rank data loaded successfully!")


def page_rank(ids, with_id=False):
    """
    Returns the page ranks values for each of the provided wiki article.

    Args:
    -----
        ids: list of ints
            list of wiki article IDs

    Returns:
    --------
        list of ints:
            list of page rank score that correspond to the provided list article IDs.
    """
    # TODO test if this works as expected
    res = []
    for pid in ids:
        try:
            res.append((pid, pr_dict[pid])) if with_id else res.append(pr_dict[pid])
        except KeyError:
            res.append((pid, None)) if with_id else res.append(None)
    return res


