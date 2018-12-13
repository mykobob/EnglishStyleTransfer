from data import *
from read_bible import *
from utils import *
from main import *

import argparse

def save(kjv_path, esv_path, kjv_data_path, esv_data_path):
    kjv, esv = load_bibles(kjv_path, esv_path, 'all')
    train_refs, dev_refs, test_refs = split_dataset(kjv, 80, 10, 10)
    with open('{}_train.csv'.format(kjv_data_path), "w") as f:
        for ref in train_refs:
            f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
    print("Done writing kjv train refs")
    with open('{}_dev.csv'.format(kjv_data_path), "w") as f:
        for ref in dev_refs:
            f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
    print("Done writing kjv dev refs")
    with open('{}_test.csv'.format(kjv_data_path), "w") as f:
        for ref in test_refs:
            f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
    print("Done writing kjv test refs")

    with open('{}_train.csv'.format(esv_data_path), "w") as f:
        for ref in train_refs:
            f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
    print("Done writing esv train refs")
    with open('{}_dev.csv'.format(esv_data_path), "w") as f:
        for ref in dev_refs:
            f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
    print("Done writing esv dev refs")
    with open('{}_test.csv'.format(esv_data_path), "w") as f:
        for ref in test_refs:
            f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
    print("Done writing esv test refs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kjv_path', required=True)
    parser.add_argument('--esv_path', required=True)
    parser.add_argument('--kjv_data_path', required=True)
    parser.add_argument('--esv_data_path', required=True)
    args = parser.parse_args()
    save(args.kjv_path, args.esv_path, args.kjv_data_path, args.esv_data_path)
