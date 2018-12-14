import argparse
import torch.nn as nn
import os
from read_bible import read_esv
from data import *
import numpy as np

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch_targets(batch, batch_lens):
    # batch is shape [batch_size, max_len_size]
    data = batch
    target = torch.from_numpy(np.zeros_like(batch)).long()
    offset = batch[:, 1:]
    target[:, :-1] = offset
    eov_idx = indexer.index_of(EOV_SYMBOL)
    for batch_idx, batch_len in enumerate(batch_lens):
        target[batch_idx, batch_len.item() - 1] = eov_idx

    return data.to(device), target.to(device)


def batchify(data_indexed, batch_size):
    nbatch = len(data_indexed) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    batches = []
    batches_sentence_lengths = []

    for i in range(0, nbatch, batch_size):
        batch_max_len = 0        
        for sentence in data_indexed[i:i+batch_size]:
            batch_max_len = max(batch_max_len, len(sentence))
        batch = np.ones((batch_size, batch_max_len)) * indexer.index_of(PAD_SYMBOL)
        batch_sentence_len = np.zeros((batch_size), dtype=np.int32)
        
        for batch_idx, sentence in enumerate(data_indexed[i:i+batch_size]):
            batch[batch_idx, :len(sentence)] = sentence
            batch_sentence_len[batch_idx] = len(sentence)
        batches.append(torch.from_numpy(batch).long())
        batches_sentence_lengths.append(torch.from_numpy(batch_sentence_len))
    return batches, batches_sentence_lengths

def evaluate(data_source, data_source_lens):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(indexer)
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_source):
            data, targets = get_batch_targets(batch, data_source_lens[batch_idx])
            output, hidden = model(data, data_source_lens[batch_idx], hidden)

            output_flat = torch.exp(output).view(-1, ntokens)
            targets_flat = targets.view(-1)
            #import pdb; pdb.set_trace()
            total_loss += len(data) * criterion(output_flat, targets_flat).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)



parser = argparse.ArgumentParser()
parser.add_argument('--lm_type', type=str, required=True)
parser.add_argument('--lm_path', type=str, required=True)
parser.add_argument('--category', required=True)
parser.add_argument('--data', required=True)

args = parser.parse_args()

if args.lm_type == 'rnn':
    # assume RNN LM
    model = torch.load(args.lm_path)
    esv = read_esv(args.data, args.category)
    
    train_path = os.path.join('data', os.path.join(args.category, f'esv_{args.category}_train.csv'))
    dev_path = os.path.join('data', os.path.join(args.category, f'esv_{args.category}_dev.csv'))
    test_path = os.path.join('data', os.path.join(args.category, f'esv_{args.category}_test.csv'))
    
    train, dev, test = load_datasets(train_path, dev_path, test_path, esv)
    train_data_indexed, dev_data_indexed, test_data_indexed, indexer = index_dataset(esv, train, dev, test, 1)
    
    train_data_indexed.sort(key=lambda data: len(data), reverse=True)
    dev_data_indexed.sort(key=lambda data: len(data), reverse=True)
    test_data_indexed.sort(key=lambda data: len(data), reverse=True)
    
    eval_batch_size = 1
    train_data, train_data_lens = batchify(train_data_indexed, eval_batch_size)
    val_data, val_data_lens = batchify(dev_data_indexed, eval_batch_size)
    test_data, test_data_lens = batchify(test_data_indexed, eval_batch_size)
    train_ce_error = evaluate(train_data, train_data_lens)
    val_ce_error = evaluate(val_data, val_data_lens)
    test_ce_error = evaluate(test_data, test_data_lens)
    #print('Train Perplexity: ', 2 ** (train_ce_error))
    #print('Val Perplexity: ', 2 ** (val_ce_error))
    print('Test Perplexity: ', 2 ** (test_ce_error))
    print('Indexed words', len(indexer))
else:
    model = get_kenlm(args.lm_path)
    esv = read_esv(args.data, args.category)
    train_path = os.path.join('data', os.path.join(args.category, f'esv_{args.category}_train.csv'))
    dev_path = os.path.join('data', os.path.join(args.category, f'esv_{args.category}_dev.csv'))
    test_path = os.path.join('data', os.path.join(args.category, f'esv_{args.category}_test.csv'))
    
    train, dev, test = load_datasets(train_path, dev_path, test_path, esv)
    train_data_indexed, dev_data_indexed, test_data_indexed, indexer = index_dataset(esv, train, dev, test, 1)
    
    test_data_indexed.sort(key=lambda data: len(data), reverse=True)
    
    eval_batch_size = 1
    test_data, test_data_lens = batchify(test_data_indexed, 1)
    all_loss = 0
    for i in range(len(test_data)):
        scores = kenlm_decode_dist(test_data[i], indexer, model)
        #scores = torch.exp(scores)
        for j in range(test_data_lens[i]) - 1:
            next_word = j + 1
            nword_idx = test_data[i][next_word]
            all_loss += nword_idx

    print('Test Perplexity: ', 2 ** (all_loss / len(test_data)))
    print('Indexed words', len(indexer))
