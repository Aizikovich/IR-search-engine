import pickle
from operator import itemgetter
from pathlib import Path
from itertools import islice, count, groupby

from collections import defaultdict, Counter
from contextlib import closing

from src.methods.tokenizer import tokenize

# from src.invertedIndex import InvertedIndex
BLOCK_SIZE = 1999998


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}{i:03}.bin', 'wb')
                          for i in count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def exit(self, exc_type, exc_value, traceback):
        self.close()
        return False


TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage
            side-effects).
        """
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            # here ok!

            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write(self, base_dir, name):
        """ Write the in-memory index to disk and populate the `posting_locs`
            variables with information about file location and offset of posting
            lists. Results in at least two files:
            (1) posting files `name`XXX.bin containing the posting lists.
            (2) `name`.pkl containing the global term stats (e.g. df).
        """

        self.posting_locs = defaultdict(list)
        with closing(MultiFileWriter(base_dir, name)) as writer:
            # iterate over posting lists in lexicographic order
            for w in sorted(self._posting_list.keys()):
                self._write_a_posting_list(w, writer, sort=True)
        # GLOBAL DICTIONARIES ############################################
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def _write_a_posting_list(self, w, writer, sort=False):
        # TODO upload to bucket instead of local file - use MultiFileWriter
        #
        # sort the posting list by doc_id
        pl = self._posting_list[w]
        if sort:
            pl = sorted(pl, key=itemgetter(0))
        # convert to bytes
        b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                      for doc_id, tf in pl])

        # write to file(s)
        locs = writer.write(b)
        # save file locations to index
        self.posting_locs[w].extend(locs)

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                # read a certain number of bytes into variable b
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)

                # int_vals = [x for x in b]
                # appearance = [int_vals[x:x + 6] for x in range(0, len(int_vals), 6)]
                # posting_list = [(x[-3], x[-1]) for x in appearance]
                #
                # # convert the bytes read into `b` to a proper posting list.
                # # YOUR CODE HERE
                posting_list = []
                j = TUPLE_SIZE
                # TODO check if the loop is correct
                while j <= self.df[w] * TUPLE_SIZE:
                    tempId = int.from_bytes(b[j - TUPLE_SIZE: j - TUPLE_SIZE + 4], 'big')
                    tempTf = int.from_bytes(b[j - TUPLE_SIZE + 4: j], 'big')
                    posting_list.append((tempId, tempTf))
                    j += TUPLE_SIZE
                if w == '#':
                    print(w, posting_list[:10])
                yield w, posting_list

    def merge_indices(self, base_dir, names, output_name):
        """ A function that merges the (partial) indices built from subsets of
            documents, and writes out the merged posting lists.
        Parameters:
        -----------
            base_dir: str
                Directory where partial indices reside.
            names: list of str
                A list of index names to merge.
            output_name: str
                The name of the merged index.
        """
        indices = [InvertedIndex.read_index(base_dir, name) for name in names]
        iters = [idx.posting_lists_iter() for idx in indices]

        self.posting_locs = defaultdict(list)

        for i in iters:
            for w, p in i:
                self._posting_list[w] += p

        for index in indices:
            self.df += index.df
            self.term_total += index.term_total

        self.write(base_dir, name=output_name)

        # GLOBAL DICTIONARIES
        self._write_globals(base_dir, output_name)

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()


# default tokenizer keeping all words
default_tokenizer = lambda text: tokenize(text)


# define batch iterator
def batch_iterator(it, batch_size=1000):
    """ Generator that yields items in a batch. Yields the batch
      index (0, 1, ..) and an iterable of items in each batch of `batch_size`
      in length.
  """
    for i, group in groupby(enumerate(it),
                            lambda x: x[0] // batch_size):
        _, batch = zip(*group)
        yield i, batch


def process_wiki(pages, index_name, tokenize_func=default_tokenizer):
    """ Process wikipedia: tokenize article body, title, anchor text, and create
      indices for them. Each index is named `index_name` and placed in a
      directory under the current dir named 'body_indices', 'title_index'
      and 'anchor_index', respectively.
  Parameters:
  -----------
  pages: list of tuples
    Each tuple is a wiki article with id, title, body, and
    [(target_article_id, anchor_text), ...].
  index_name: str
    The name for the index.
  tokenize_func: function str -> list of str
    Tokenization function that takes text as input and return a list of
    tokens.
  Returns:
  --------
  Three inverted index objects
    body_index, title_index, anchor_index.
  """
    # create the index for titles
    title_index = InvertedIndex()
    # collect anchor text tokens for each target article by its id
    id2anchor_text = defaultdict(list)

    # iterate over batches of pages from the dump
    body_index_names = []
    for batch_idx, batch_pages in batch_iterator(pages):
        ids, titles, bodies, links = zip(*batch_pages)
        target_ids, anchor_texts = zip(*[wl for l in links for wl in l])
        # tokenize
        titles = map(tokenize_func, titles)
        bodies = map(tokenize_func, bodies)
        anchor_texts = map(tokenize_func, anchor_texts)
        # create a separate index of articles body for article in this batch
        body_index = InvertedIndex()
        for id, title, body in zip(ids, titles, bodies):
            title_index.add_doc(id, title)
            body_index.add_doc(id, body)
        for target_id, anchor_text in zip(target_ids, anchor_texts):
            id2anchor_text[target_id].extend(anchor_text)
        body_index.write('./body_indices', f'{index_name}_{batch_idx}')
        body_index_names.append(f'{index_name}_{batch_idx}')
    # merge body indices from the different batches into one index and delete
    # the parts
    body_index = InvertedIndex()
    body_index.merge_indices('./body_indices',
                             body_index_names, index_name)
    for idx_name in body_index_names:
        InvertedIndex.delete_index('./body_indices', idx_name)
    title_index.write('./title_index', index_name)
    # create index for anchor text
    anchor_index = InvertedIndex()
    for id, tokens in id2anchor_text.items():
        anchor_index.add_doc(id, tokens)
    anchor_index.write('./anchor_index', index_name)
    return body_index, title_index, anchor_index


def main():
    # read the dump
    pkl_file = "part15_preprocessed.pkl"
    with open(pkl_file, 'rb') as f:
        pages = pickle.load(f)
        # process the dump
    body_ix, title_ix, anchor_ix = process_wiki(pages, 'wiki_index')
    print(body_ix)


if __name__ == '__main__':
    main()
