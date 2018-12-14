#! /bin/bash

python main.py --train_path data/luke_to_acts/kjv_luke_to_acts_train.csv --dev_path data/luke_to_acts/kjv_luke_to_acts_dev.csv --test_path data/luke_to_acts/kjv_luke_to_acts_test.csv --esv data/esv.txt --kjv data/kjv.csv --epochs 10 --lr 0.0004 --hidden_size 512 --category luke_acts --gpu 0 --model_path luke_acts.pt --indexer_path esv_indexer.obj --gpu 0 > results/kenlm/luke_acts &

python main.py --train_path data/gospels/kjv_gospels_train.csv --dev_path data/gospels/kjv_gospels_dev.csv --test_path data/gospels/kjv_gospels_test.csv --esv data/esv.txt --kjv data/kjv.csv --epochs 10 --lr 0.0004 --hidden_size 512 --category gospels --gpu 1 --learn_weights True > results/kenlm/gospels_learn &

python main.py --train_path data/gospels/kjv_gospels_train.csv --dev_path data/gospels/kjv_gospels_dev.csv --test_path data/gospels/kjv_gospels_test.csv --esv data/esv.txt --kjv data/kjv.csv --epochs 10 --lr 0.0004 --hidden_size 512 --category gospels --gpu 2 > results/kenlm/gospels &
