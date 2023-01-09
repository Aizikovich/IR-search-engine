import pickle
import re

import csv
import pandas as pd
import requests


# Todo: in the small corpus we are missing the anchor wikipedia pages
def get_page_titles(id_lis):
    """ Returns the title of the first, fourth, and fifth pages as ranked about 
      by PageRank.
    Returns:
    --------
    list of three strings.
    """
    # YOUR CODE HERE
    j = 0
    titles = []
    for i in range(len(id_lis)):
        title = requests.get('https://en.wikipedia.org/?curid=' + str(id_lis[i])).text
        title = re.findall('<title>(.*?)</title>', title)[0]
        title = title.replace(' - Wikipedia', '')
        titles.append(title)
        if j % 5 == 0:
            print(j)
        j += 1
    return titles


if __name__ == "__main__":
    pkl_file = "part15_preprocessed.pkl"
    with open(pkl_file, 'rb') as f:
        pages = pickle.load(f)

    pages_id = [page[0] for page in pages]
    names = get_page_titles(pages_id)

    with open('title_names.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(zip(pages_id, names))
