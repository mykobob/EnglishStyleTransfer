#! /bin/bash

python main.py --train_path data/epistles/kjv_epistles_train.csv --dev_path data/epistles/kjv_epistles_dev.csv --test_path data/epistles/kjv_epistles_test.csv --esv data/esv.txt --kjv data/kjv.csv --epochs 10 --lr 0.0004 --hidden_size 512 --category epistles --gpu 0 --learn_weights True >  results/kenlm/epistles_learn &

python main.py --train_path data/epistles/kjv_epistles_train.csv --dev_path data/epistles/kjv_epistles_dev.csv --test_path data/epistles/kjv_epistles_test.csv --esv data/esv.txt --kjv data/kjv.csv --epochs 10 --lr 0.0004 --hidden_size 512 --category epistles --gpu 1 >  results/kenlm/epistles &

python main.py --train_path data/luke_to_acts/kjv_luke_to_acts_train.csv --dev_path data/luke_to_acts/kjv_luke_to_acts_dev.csv --test_path data/luke_to_acts/kjv_luke_to_acts_test.csv --esv data/esv.txt --kjv data/kjv.csv --epochs 10 --lr 0.0004 --hidden_size 512 --category luke_acts --gpu 0 --model_path luke_acts.pt --indexer_path esv_indexer.obj --gpu 2 --learn_weights True > results/kenlm/luke_acts_learn &
