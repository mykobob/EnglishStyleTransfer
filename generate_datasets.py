from data import *
from read_bible import *
from utils import *
from main import *

kjv, esv = load_bibles("data/kjv.csv", "data/esv.csv")
train_refs, dev_refs, test_refs = split_dataset(kjv, 80, 10, 10)
with open("kjv_train.csv") as f:
    for ref in train_refs:
        f.write("%s,%s,%s" % ref[0], ref[1], ref[2])
with open("kjv_dev.csv") as f:
    for ref in dev_refs:
        f.write("%s,%s,%s" % ref[0], ref[1], ref[2])
with open("kjv_test.csv") as f:
    for ref in test_refs:
        f.write("%s,%s,%s" % ref[0], ref[1], ref[2])

with open("esv_train.csv") as f:
    for ref in train_refs:
        f.write("%s,%s,%s" % ref[0], ref[1], ref[2])
with open("esv_dev.csv") as f:
    for ref in dev_refs:
        f.write("%s,%s,%s" % ref[0], ref[1], ref[2])
with open("esv_test.csv") as f:
    for ref in test_refs:
        f.write("%s,%s,%s" % ref[0], ref[1], ref[2])