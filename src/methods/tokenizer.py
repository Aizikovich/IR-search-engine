import re
import nltk
nltk.download('stopwords')
stopwords_frozen = frozenset(nltk.corpus.stopwords.words('english'))

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)


def tokenize(text):
    """
    The function takes a text, lowercases it, and returns a list of tokens if token not in stopwords
    :param text: str - text to tokenize
    :return: list - of tokens
    """
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in stopwords_frozen and token.group() not in corpus_stopwords]
    return list_of_tokens




stemmer = nltk.PorterStemmer()


# Getting tokens from the text while removing punctuations.
def filter_tokens(tokens, tokens2remove=None, use_stemming=False):
    """
    The function takes a list of tokens, filters out `tokens2remove` and
      stem the tokens using `stemmer`.
    Parameters:
    -----------
    tokens: list of str.
    Input tokens.
    tokens2remove: frozenset.
    Tokens to remove (before stemming).
    use_stemming: bool.
    If true, apply stemmer.stem on tokens.
    Returns:
    --------
    list of tokens from the text.
    """
    # YOUR CODE HERE
    if tokens2remove is not None:
        tokens = [x for x in tokens if x not in tokens2remove]
    if use_stemming == True:
        tokens = [stemmer.stem(w) for w in tokens]
    else:
        return tokens
    return tokens


def get_pls_tokens(index, tokens):
    """
    the function takes a list of tokens and returns a list of tokens that are in the index
    :param index: InvertedIndex object
    :param tokens: tokens to filter
    :return: list of tokens that are in the index
    """
    return index.postings_list(tokens)

