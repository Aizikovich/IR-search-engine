import pandas as pd


def normalize(df):
    """
    This function normalizes the given df.
    Parameters:
    -----------
    df: pandas df. The df to normalize. It should have two columns: id and score.
    Returns:
    -----------
    pandas df. The normalized df.
    """
    # Get the max score
    max_score = df.max()
    # get the min score
    min_score = df.min()
    # Normalize the scores
    df = (df - min_score) / (max_score - min_score)
    # Return the normalized df
    return df


def main_search(title_res, body_res, anchor_res, page_rank, page_view, wight_d=(1, 1, 1, 1, 1)):
    """
    This function is the main search function. It takes the results of the all search methods and combines them to one
    result.
    It normalizes the results and then combines them to one result.
    Parameters:
    -----------
    titleRes: list of tuples. Each tuple is a pair of document id and score. For example:
                                                                                     [
                                                                                         (1, 0.5),
                                                                                         (2, 0.3),
                                                                                         (3, 0.1)
                                                                                     ]
    bodyRes: list of tuples. Each tuple is a pair of document id and score.
    anchorRes: list of tuples. Each tuple is a pair of document id and score.
    pageRank: list of tuples. Each tuple is a pair of document id and score.
    pageView: list of tuples. Each tuple is a pair of document id and score.
    Returns:
    -----------
    list of ints. sorted by their combined score

    """
    # Combine the results into one df by id
    df = pd.DataFrame(title_res, columns=['id', 'title'])

    # add body res, add new ids if needed
    df = df.merge(pd.DataFrame(body_res, columns=['id', 'body']), how='outer', on='id')
    df = df.merge(pd.DataFrame(anchor_res, columns=['id', 'anchor']), how='outer', on='id')
    df = df.merge(pd.DataFrame(page_rank, columns=['id', 'page_rank']), how='outer', on='id')
    df = df.merge(pd.DataFrame(page_view, columns=['id', 'page_view']), how='outer', on='id')
    # Fill NaN with 0
    df = df.fillna(0)
    df = df.set_index('id')
    # print(df)
    # apply normalization function to all columns except id
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(normalize, axis=0)

    # set id as index
    # Calculate the combined score
    df['score'] = df['title'] * wight_d[0] + \
                  df['body'] * wight_d[1] + \
                  df['anchor'] * wight_d[2] + \
                  df['page_rank'] * wight_d[3] + \
                  df['page_rank'] * wight_d[4]
    # Sort the results by the combined score
    df = df.sort_values(by=['score'], ascending=False)
    # Return the sorted list of ids
    print(df)
    return df.index.tolist()[:100]


# test
if __name__ == '__main__':
    title = [(1, 0.5), (2, 0.0), (5, 0.1)]
    body = [(1, 0.5), (2, 0.3), (3, 1)]
    anchor = [(1, 0.5), (2, 0.3), (4, 0.1)]
    pagerank = [(1, 155), (2, 0.3), (6, 0.8)]
    pageview = [(1, 5454), (7, 30), (3, 454541)]
    print(main_search(title, body, anchor, pagerank, pageview))
