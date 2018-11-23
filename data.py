from utils import *

# Do we need a wrapper class for each dataset? Or just tokenized strings

# Might still need these tokens
PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"


# Reads a dataset in from the given file
def load_dataset(filename):
    dataset = []
    with open(filename) as f:
        for line in f:
            # Figure out how to split sentences/verses
            x, y = line.rstrip('\n').split('\t')
            dataset.append((x, y))
            # For copying task
            # dataset.append((x, x))
    return dataset