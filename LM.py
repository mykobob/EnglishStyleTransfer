# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np

from torch import optim
import pickle
from read_bible import read_esv
from data import load_datasets, index_dataset, PAD_SYMBOL, EOV_SYMBOL
from LM_models import RNNLM, FFNNLM

parser = argparse.ArgumentParser(description='ESV Language Model')
parser.add_argument('--data', type=str, default='./data/esv.txt',
                    help='location of the data corpus')
parser.add_argument('--indexer_file', type=str, default='esv_indexer.obj',
                    help='location to store indexer')
parser.add_argument('--translation', type=str, default='esv',
                    help='What translation to train on')
parser.add_argument('--data_type', type=str, default='gospels',
                    help='What books to train on')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (GRU, FFNN)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='best_model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()
print(args)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

esv = read_esv(args.data, args.data_type)
# import pdb; pdb.set_trace()
train_path = os.path.join('data', os.path.join(args.data_type, f'{args.translation}_{args.data_type}_train.csv'))
dev_path = os.path.join('data', os.path.join(args.data_type, f'{args.translation}_{args.data_type}_dev.csv'))
test_path = os.path.join('data', os.path.join(args.data_type, f'{args.translation}_{args.data_type}_test.csv'))
train, dev, test = load_datasets(train_path, dev_path, test_path, esv)
train_data_indexed, dev_data_indexed, test_data_indexed, indexer = index_dataset(esv, train, dev, test, args.bptt)

train_data_indexed.sort(key=lambda data: len(data), reverse=True)
dev_data_indexed.sort(key=lambda data: len(data), reverse=True)
test_data_indexed.sort(key=lambda data: len(data), reverse=True)

# corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data_indexed, batch_size):
#     import pdb; pdb.set_trace()
    # Work out how cleanly we can divide the dataset into bsz parts.
#     nbatch = data.size() // batch_size
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
#     data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
#     data = data.view(batch_size, -1).t().contiguous()
    return batches, batches_sentence_lengths

eval_batch_size = 1
train_data, train_data_lens = batchify(train_data_indexed, args.batch_size)
val_data, val_data_lens = batchify(dev_data_indexed, eval_batch_size)
test_data, test_data_lens = batchify(test_data_indexed, eval_batch_size)


###############################################################################
# Build the model
###############################################################################

ntokens = len(indexer)

if args.model == 'GRU':
    model = RNNLM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, True, args.dropout, args.tied).to(device)
elif args.model == 'FFNN':
    model = FFNNLM(args.window_size, args.nhid, args.emsize, indexer)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch_targets(batch, batch_lens):
    # batch is shape [batch_size, max_len_size]
#     seq_len = min(args.bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
    data = batch
    target = torch.from_numpy(np.zeros_like(batch)).long()
    offset = batch[:, 1:]
    target[:, :-1] = offset
    eov_idx = indexer.index_of(EOV_SYMBOL)
    for batch_idx, batch_len in enumerate(batch_lens):
        target[batch_idx, batch_len.item() - 1] = eov_idx

    return data.to(device), target.to(device)


def evaluate(data_source, data_source_lens):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(indexer)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
#         for i in range(0, data_source.size(0) - 1, args.bptt):
#         import pdb; pdb.set_trace()
        for batch_idx, batch in enumerate(data_source):
#             data, targets = get_batch(data_source, i)
            data, targets = get_batch_targets(batch, data_source_lens[batch_idx])
#             output, hidden = model(data, hidden)
            output, hidden = model(data, data_source_lens[batch_idx], hidden)

            output_flat = output.view(-1, ntokens)
            targets_flat = targets.view(-1)
            total_loss += len(data) * criterion(output_flat, targets_flat).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(indexer)
    hidden = model.init_hidden(args.batch_size)
    optimizer = optim.Adam(list(model.parameters()), lr=lr)
    for batch_idx, batch in enumerate(train_data):
        data, targets = get_batch_targets(batch, train_data_lens[batch_idx])
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, train_data_lens[batch_idx], hidden)
        model_flat = output.view(-1, ntokens)
        targets_flat = targets.view(-1)
        loss = criterion(model_flat, targets_flat)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_idx, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Save indexer so we can load it for the LSTM
with open(args.indexer_file, 'wb') as f:
    pickle.dump(indexer, f)
    
# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data, val_data_lens)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            print('Found new best model')
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data, test_data_lens)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

# import pdb; pdb.set_trace()
sentence = 'For God so loved the world'.split(' ')
sentence_idx = torch.tensor([[indexer.index_of(token) for token in sentence]]).to(device)
sentence_lens = torch.tensor(len(sentence)).unsqueeze(0).to(device)
# hidden = model.hidden
hidden = model.init_hidden(1)
output, hidden = model(sentence_idx, sentence_lens, hidden)
for i in range(len(sentence)):
    print('Input:', sentence[i])
    if i != len(sentence) - 1:
        print('Expected:', sentence[i+1], '. Predicted:', indexer.get_object(torch.argmax(output[0][i]).item()))
    else:
        print('Expected: <EOV>. Predicted:', indexer.get_object(torch.argmax(output[0][i]).item()))
    print()

import pdb; pdb.set_trace()

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
