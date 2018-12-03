from data import *
from read_bible import *
from utils import *
from main import *

kjv, esv = load_bibles("data/kjv.csv", "data/esv.txt")
train_refs, dev_refs, test_refs = split_dataset(kjv, 80, 10, 10)
with open("data/kjv_train.csv", "w") as f:
    for ref in train_refs:
        f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
print("Done writing kjv train refs")
with open("data/kjv_dev.csv", "w") as f:
    for ref in dev_refs:
        f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
print("Done writing kjv dev refs")
with open("data/kjv_test.csv", "w") as f:
    for ref in test_refs:
        f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
print("Done writing kjv test refs")

with open("data/esv_train.csv", "w") as f:
    for ref in train_refs:
        f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
print("Done writing esv train refs")
with open("data/esv_dev.csv", "w") as f:
    for ref in dev_refs:
        f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
print("Done writing esv dev refs")
with open("data/esv_test.csv", "w") as f:
    for ref in test_refs:
        f.write("%s,%s,%s\n" % (ref[0], ref[1], ref[2]))
print("Done writing esv test refs")
