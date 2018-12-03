from utils import *
import random

from read_bible import *

# Wrapper class for an example.
# x = the natural language as one string
# x_tok = tokenized NL, a list of strings
# x_indexed = indexed tokens, a list of ints
# y = the logical form
# y_tok = tokenized logical form, a list of strings
# y_indexed = indexed logical form
class Example(object):
    def __init__(self, x, x_tok, x_indexed, y, y_tok, y_indexed):
        self.x = x
        self.x_tok = x_tok
        self.x_indexed = x_indexed
        self.y = y
        self.y_tok = y_tok
        self.y_indexed = y_indexed

    def __repr__(self):
        return " ".join(self.x_tok) + " => " + " ".join(self.y_tok) + "\n   indexed as: " + repr(self.x_indexed) + " => " + repr(self.y_indexed)

    def __str__(self):
        return self.__repr__()


# Wrapper for a Derivation consisting of an Example object, a score/probability associated with that example,
# and the tokenized prediction.
class Derivation(object):
    def __init__(self, example, p, y_toks):
        self.example = example
        self.p = p
        self.y_toks = y_toks

    def __str__(self):
        return "%s (%s)" % (self.y_toks, self.p)

    def __repr__(self):
        return self.__str__()


PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"
SOV_SYMBOL = "<SOV>"
EOV_SYMBOL = "<EOV>"


def load_bibles(kjv_path, esv_path):
    all_kjv = read_kjv(kjv_path)
    all_esv = read_esv(esv_path)
    return all_kjv, all_esv

# Reads the training, dev, and test data from the corresponding files.
def load_datasets(train_path, dev_path, test_path, bible):
    train_verses = load_dataset(train_path, bible)
    dev_verses = load_dataset(dev_path, bible)
    test_verses = load_dataset(test_path, bible)
    return train_verses, dev_verses, test_verses

# takes a bible and path to file with references and returns a list of tokenized verses
def load_dataset(file_path, bible):
    import csv
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        verses = list(reader)
    for i in len(verses):
        verses[i][1] = int(verses[i][1])
        verses[i][2] = int(verses[i][2])
    return verses

# Whitespace tokenization
def tokenize(x):
    return x.split()


def index(x_tok, indexer):
    return [indexer.index_of(xi) if indexer.index_of(xi) >= 0 else indexer.index_of(UNK_SYMBOL) for xi in x_tok]


def index_data(src_text, dest_text, input_indexer, output_indexer, example_len_limit):
    data_indexed = []
    for book_name, chapter_num, verse_num in data:
        x_tok = src_text[book_name][chapter_num][verse_num]
        y_tok = dest_text[book_name][chapter_num][verse_num][0:example_len_limit]
        data_indexed.append(Example(' '.join(x_tok), x_tok, index(x_tok, input_indexer), ' '.join(y_tok), y_tok,
                                          index(y_tok, output_indexer)))
    return data_indexed


# Indexes train and test datasets where all words occurring less than or equal to unk_threshold times are
# replaced by UNK tokens.

# Input is 
    # KJV/ESV dicts of {book_name -> {chapter_num -> {verse_num: -> [character tokens]}}}
    # train/dev/test set information
# Output should be 
    # a list of Example objects for train/dev/test
def index_datasets(src_text, dest_text, train, dev, test, example_len_limit, unk_threshold=0.0):
    input_word_counts = Counter()
    # Count words and build the indexers
    for book_name, chapter_num, verse_num in train:
        for token in src_text[book_name][chapter_num][verse_num]:
            input_word_counts.increment_count(token, 1.0)
            
    input_indexer = Indexer()
    output_indexer = Indexer()
    # Reserve 0 for the pad symbol for convenience
    input_indexer.get_index(PAD_SYMBOL)
    input_indexer.get_index(UNK_SYMBOL)
    output_indexer.get_index(PAD_SYMBOL)
    output_indexer.get_index(SOV_SYMBOL)
    output_indexer.get_index(EOV_SYMBOL)
    # Index all input words above the UNK threshold
    for word in input_word_counts.keys():
        if input_word_counts.get_count(word) > unk_threshold + 0.5:
            input_indexer.get_index(word)
    # Index all output tokens in train
    for book_name, chapter_num, verse_num in train:
        for token in dest_text[book_name][chapter_num][verse_num]:
            output_indexer.get_index(token)
    # Index things
    train_data_indexed = index_data(train, input_indexer, output_indexer, example_len_limit)
    dev_data_indexed = index_data(dev, input_indexer, output_indexer, example_len_limit)
    test_data_indexed = index_data(test, input_indexer, output_indexer, example_len_limit)
    return train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer