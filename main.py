import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *


def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true',
                        help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv',
                        help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')

    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--input_dim', type=int, default=100, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=100, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')

    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false',
                        help='bidirectional LSTM')
    parser.add_argument('--no_reverse_input', dest='reverse_input', default=True, action='store_false',
                        help='disable_input_reversal')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='input dropout rate')
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help='dropout rate internal to encoder RNN')
    args = parser.parse_args()
    return args


# Do we need to update this?
class Seq2SeqSemanticParser(object):
    def __init__(self, enc, dec, input_emb, output_emb, in_indexer, out_indexer):
        # Add any args you need here
        self.enc = enc
        self.dec = dec
        self.input_emb = input_emb
        self.output_emb = output_emb
        self.in_indexer = in_indexer
        self.out_indexer = out_indexer

    def decode(self, test_data):
        self.enc.eval()
        self.dec.eval()
        self.input_emb.eval()
        self.output_emb.eval()
        with torch.no_grad():
            test_derivs = []
            correct = 0
            for test_ex in test_data:
                token = torch.tensor((self.out_indexer.index_of(SOS_SYMBOL))).unsqueeze(0)
                _, _, hidden = encode_input_for_decoder(torch.tensor(test_ex.x_indexed).unsqueeze(0),
                                                        torch.tensor(len(test_ex.x_indexed)).unsqueeze(0),
                                                        self.input_emb, self.enc)
                this_deriv = []
                # if (method2):
                #     counter = 0
                #     while token != self.out_indexer.index_of(EOS_SYMBOL):
                #         word_emb = self.input_emb.forward(token).unsqueeze(0)
                #         word_prob, hidden = self.dec(word_emb, hidden)
                #         if counter > 65:
                #             break
                #         prediction = torch.argmax(word_prob, dim=1)
                #         this_deriv.append(prediction)
                #         counter += 1
                #         token = prediction

                # max_len = len(test_ex.y_tok)
                max_len = 65
                counter = 0
                while counter < max_len:
                    output = self.output_emb.forward(token).unsqueeze(0)
                    output, hidden = self.dec.forward(output, hidden)
                    token = torch.argmax(output, dim=1)
                    if token.item() == self.out_indexer.index_of(EOS_SYMBOL):
                        break
                    this_deriv.append(token)
                    counter += 1
                this_deriv = [self.out_indexer.get_object(x.item()) for x in this_deriv]
                for k in range(min(len(test_ex.y_tok), len(this_deriv))):
                    if test_ex.y_tok[k] == this_deriv[k]:
                        correct += 1
                test_derivs.append([Derivation(test_ex, 1.0, this_deriv)])
        print("Correct in decode:", correct)
        return test_derivs


# Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
# Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])


# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array(
        [[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)]
         for ex in exs])


# Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
# inp_lens_tensor lengths.
# x_tensor: batch size x sent len tensor of input token indices
# inp_lens: batch size length vector containing the length of each sentence in the batch
# model_input_emb: EmbeddingLayer
# model_enc: RNNEncoder
# Returns the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting

# E.g., calling this with x_tensor (0 is pad token):
# [[12, 25, 0, 0],
#  [1, 2, 3, 0],
#  [2, 0, 0, 0]]
# inp_lens = [2, 3, 1]
# will return outputs with the following shape:
# enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
# enc_final_states = 3 x dim
def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


# Need to update
def train_model_encdec(train_data, test_data, input_indexer, output_indexer, args):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    input_lens = torch.from_numpy(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    # Create indexed output
    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    output_lens = torch.from_numpy(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Create models
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_dec = RNNDecoder(args.output_dim, args.hidden_size, len(output_indexer))

    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters
    print("\n")
    params = list(model_enc.parameters()) + list(model_dec.parameters()) + \
             list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    opt = optim.Adam(params, lr=args.lr)
    loss_function = torch.nn.NLLLoss()

    for epoch in range(0, args.epochs):
        total_loss = 0
        model_enc.train()
        model_dec.train()
        model_input_emb.train()
        model_output_emb.train()
        total_correct = 0
        for idx in range(0, all_train_input_data.shape[0]):
            opt.zero_grad()
            token = torch.tensor([output_indexer.index_of(SOS_SYMBOL)])
            prediction = []
            _, _, hidden = encode_input_for_decoder(torch.from_numpy(all_train_output_data)[idx].unsqueeze(0),
                                                    input_lens[idx].unsqueeze(0),
                                                    model_input_emb, model_enc)
            labels = torch.from_numpy(all_train_output_data)[idx]
            # Probability matrix
            outputs = torch.zeros((output_lens[idx], len(output_indexer)))
            for output_idx in range(0, output_lens[idx]):
                if output_idx == 0:
                    token = torch.tensor((output_indexer.index_of("<SOS>")))
                    token = model_output_emb(token).unsqueeze(0).unsqueeze(0)
                    word_prob, hidden = model_dec(token, hidden)
                else:
                    token = torch.tensor(train_data[idx].y_indexed[output_idx - 1])
                    token = model_output_emb(token).unsqueeze(0).unsqueeze(0)
                    word_prob, hidden = model_dec(token, hidden)
                prediction.append(torch.argmax(word_prob, dim=1).item())
                outputs[output_idx, :] = torch.squeeze(word_prob)
            # if (method2):
            #     for output_idx in range(0, output_lens[idx]):
            #         output = model_output_emb.forward(token).unsqueeze(0)
            #         output, hidden = model_dec(output, hidden)
            #         outputs[output_idx, :] = torch.squeeze(output)
            #         prediction.append(torch.argmax(output, dim=1).item())
            #         # teacher
            #         token = all_train_output_data[idx][output_idx]
            #         token = torch.LongTensor([token])
            targets = labels[:output_lens[idx]]
            # bugfixing
            for k in range(len(targets)):
                if prediction[k] == targets[k]:
                    total_correct += 1
            loss = loss_function(outputs, targets)

            total_loss += loss.item()
            loss.backward()
            opt.step()
        print("Loss on epoch %i: %f" % (epoch + 1, total_loss))
        print("Total correct predictions", total_correct)
        tempseq2seq = Seq2SeqSemanticParser(model_enc, model_dec, model_input_emb, model_output_emb, input_indexer,
                                            output_indexer)
        evaluate(test_data, tempseq2seq, example_freq=25, print_output=True)
    print("\n")
    return Seq2SeqSemanticParser(model_enc, model_dec, model_input_emb, model_output_emb, input_indexer, output_indexer)


# Not sure how we're going to score our outputs
def evaluate(test_data, decoder, example_freq=50, print_output=True, outfile=None):
    pred_derivations = decoder.decode(test_data)
    selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations)
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    for i, ex in enumerate(test_data):
        if i % example_freq == 0:
            print('Example %d' % i)
            print('  x      = "%s"' % ex.x)
            print('  y_tok  = "%s"' % ex.y_tok)
            print('  y_pred = "%s"' % selected_derivs[i].y_toks)
        # Compute accuracy metrics
        y_pred = ' '.join(selected_derivs[i].y_toks)
        # Check exact match
        if y_pred == ' '.join(ex.y_tok):
            num_exact_match += 1
        # Check position-by-position token correctness
        num_tokens_correct += sum(a == b for a, b in zip(selected_derivs[i].y_toks, ex.y_tok))
        total_tokens += len(ex.y_tok)
        # Check correctness of the denotation
        if denotation_correct[i]:
            num_denotation_match += 1
    if print_output:
        print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
        print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
        print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))
        print("\n")
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()


def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer) / denom)


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev,
                                                                                                            test,
                                                                                                            args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (
    len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    # print("Here are some examples post tokenization and indexing:")
    # for i in range(0, min(len(train_data_indexed), 10)):
    #     print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:
        decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv")


