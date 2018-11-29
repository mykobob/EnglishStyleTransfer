import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from models import *
from data import *
from read_kjv import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device('cpu')

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

    parser.add_argument('--reverse_input', type=bool, default=False)
    parser.add_argument('--emb_dropout', type=float, default=0.2)
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--bidirectional', type=bool, default=False)

    args = parser.parse_args()
    return args


class Seq2SeqSemanticParser(object):
    def __init__(self, encoder, input_embed, output_embed, decoder, input_indexer, output_indexer):
        # Add any args you need here
        self.encoder = encoder
        self.model_input_embed = input_embed
        self.model_output_embed = output_embed
        self.decoder = decoder
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer

    def start_token(self, output):
        return self.output_indexer.index_of(SOS_SYMBOL) if output else self.input_indexer.index_of(SOS_SYMBOL)

    def end_token_idx(self, output):
        return self.output_indexer.index_of(EOS_SYMBOL) if output else self.input_indexer.index_of('?')

    def not_end_of_sentence(self, token_idx, output):
        return token_idx.item() != self.end_token_idx(output)

    def get_token(self, output, token_idx):
        return self.output_indexer.get_object(token_idx) if output else self.input_indexer.get_object(token_idx)

    def toggle_decoding(self):
        self.encoder.eval()
        self.model_input_embed.eval()
        self.model_output_embed.eval()
        self.decoder.eval()

# TODO rewrite end derivation code
    def decode(self, test_data):
        # loop through test_data (maybe in batches)
        self.toggle_decoding()

        with torch.no_grad():
            ans = []
            for test_ex in test_data:
                # Encoder part
                word_indexes = torch.tensor(test_ex.x_indexed).unsqueeze(0).to(device)
                input_len = torch.tensor(len(test_ex.x_indexed)).unsqueeze(0).to(device)
                enc_out_each_word, enc_context_mask, enc_final_states \
                    = encode_input_for_decoder(word_indexes, input_len, self.model_input_embed, self.encoder)

                # Decode here
                output = True
                input_token = torch.tensor((self.start_token(output))).to(device)
                test_tokens_idx = []
                prediction_score = 0.
                output_idx = 0

                while self.not_end_of_sentence(input_token, output):
                    input_embed = prep_word_for_decoder(input_token, self.model_output_embed).unsqueeze(0).unsqueeze(0)

                    if output_idx == 0:
                        output_probs = self.decoder(enc_final_states, input_embed, enc_out_each_word)
                    else:
                        output_probs = self.decoder(self.decoder.hidden, input_embed, enc_out_each_word)

                    if output_idx < len(test_ex.y_indexed):
                        gt_word_idx = test_ex.y_indexed[output_idx]
                        # print('output_probs', output_probs.size())
                        prediction_score += output_probs[gt_word_idx].item()
                        # HOW TO CALCULATE THIS??
                    else:
                        if output_idx > len(test_ex.y_indexed) + 3:
                            break

                    prediction_idx = torch.argmax(output_probs)

                    # Feed in predicted value into next lstm cell
                    output_idx += 1
                    input_token = prediction_idx

                    test_tokens_idx.append(prediction_idx.item())
                test_tokens_idx = test_tokens_idx[:-1]
                test_tokens = [self.get_token(output, token_idx) for token_idx in test_tokens_idx]
                # ans.append([Derivation(test_ex, prediction_score, test_tokens)])

        return ans


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


def prep_word_for_decoder(output_words, model_output_emb):
    output_emb = model_output_emb.forward(output_words)
    return output_emb


def split_dataset(data, training, dev, test):
    required_verses = training + dev + test
    train_list = []
    dev_list = []
    test_list = []
    books = all_books()
    for book in books:
        for chapter in data[book]:
            counter = 0
            max_verses = len(data[book][chapter])
            if required_verses > max_verses:
                raise ValueError("not enough verses", required_verses, "in chapter", book, chapter)
            samples = random.sample(range(1, max_verses + 1), required_verses)
            for i in range(counter, counter + training):
                train_list.append((book, chapter, samples[i]))
            counter += training
            for i in range(counter, counter + dev):
                dev_list.append((book, chapter, samples[i]))
            counter += dev
            for i in range(counter, counter + test):
                test_list.append((book, chapter, samples[i]))
            counter += test
    return train_list, dev_list, test_list


def train_model_encdec(train_data, test_data, input_indexer, output_indexer, args):
    # Grab x verses from a dataset as training, x as dev, and x as test
    kjv = read_kjv("t_kjv.csv")
    num_train = 6
    num_dev = 2
    num_test = 2
    train_list, dev_list, test_list = split_dataset(kjv, num_train, num_dev, num_test)

    print(args)
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_lens = np.asarray([len(ex.x_indexed) for ex in train_data])
    input_max_len = np.max(input_lens)
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    input_lens_device = torch.from_numpy(input_lens).to(device)
    all_train_input_data_device = torch.from_numpy(all_train_input_data).to(device)
    all_test_input_data_device = torch.from_numpy(all_test_input_data).to(device)

    output_lens = np.asarray([len(ex.y_indexed) for ex in train_data])
    output_max_len = np.max(output_lens)
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    output_lens_device = torch.from_numpy(output_lens).to(device)
    all_train_output_data_device = torch.from_numpy(all_train_output_data).to(device)
    all_test_output_data_device = torch.from_numpy(all_test_output_data).to(device)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout).to(device)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional).to(device)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout).to(device)

    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters
    epochs = args.epochs
    model_dec = RNNDecoder(args.input_dim, model_enc.get_output_size(), args.hidden_size, len(output_indexer),
                           args.rnn_dropout).to(device)
    optimizer = optim.Adam(
        list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_input_emb.parameters()) + list(
            model_output_emb.parameters()), lr=args.lr)

    for epoch in range(epochs):
        model_input_emb.train()
        model_enc.train()
        model_output_emb.train()
        model_dec.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        start = time.time()
        for example in range(all_train_input_data.shape[0]):
            optimizer.zero_grad()

            # Encoder part
            word_indexes = all_train_input_data_device[example].unsqueeze(0)
            input_len = input_lens_device[example].unsqueeze(0)
            enc_output_each_word, enc_context_mask, enc_final_states_reshaped = \
                encode_input_for_decoder(word_indexes, input_len, model_input_emb, model_enc)

            # Decoder part
            expected_output = all_train_output_data_device[example]
            decoder_embeds = prep_word_for_decoder(expected_output.unsqueeze(0), model_output_emb)
            output_len = output_lens_device[example]
            output_probs = torch.zeros((output_len, len(output_indexer))).to(device)

            for i in range(output_len):
                if i == 0:
                    embedded_word = torch.tensor((output_indexer.index_of(SOS_SYMBOL))).to(device)
                    embedded_word = prep_word_for_decoder(embedded_word, model_output_emb).unsqueeze(0).unsqueeze(0)
                    token_out = model_dec(enc_final_states_reshaped, embedded_word, enc_output_each_word)
                else:
                    embedded_word = decoder_embeds[:, i - 1, :].unsqueeze(0)
                    token_out = model_dec(model_dec.hidden, embedded_word, enc_output_each_word)

                # print('token_out', token_out)
                output_probs[i, :] = token_out

            loss = torch.nn.NLLLoss()
            loss_value = loss(output_probs, expected_output[:output_len])
            total_loss += loss_value.item()

            predictions = torch.argmax(output_probs, dim=1)
            cur_correct = torch.sum(predictions == expected_output[:output_len])
            total_correct += cur_correct
            total_tokens += output_len

            loss_value.backward()
            optimizer.step()

            if example % 100 == 0:
                print('example {}/{} is done'.format(example, all_train_input_data.shape[0]))
        print(f'Epoch {epoch+1} done. Loss: {total_loss:.2f}. It took {time.time() - start} seconds')
        print(f'Total correct: {total_correct}/{total_tokens}')
        parser = Seq2SeqSemanticParser(model_enc, model_input_emb, model_output_emb, model_dec, input_indexer,
                                       output_indexer)
        evaluate(test_data, parser)
        print()
    return parser


# TODO Rewrite this to do BLEU score or compare against correct translation
def evaluate(test_data, decoder, example_freq=50, print_output=True, outfile=None):
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

# TODO rewrite loading data methods
    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev,
                                                                                                            test,
                                                                                                            args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (
    len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
    print("=======FINAL EVALUATION=======")
    # evaluate(test_data_indexed, decoder, outfile="geo_test_output.tsv")
    eval_time = time.time()
    evaluate(test_data_indexed, decoder, print_output=True, outfile="geo_test_output.tsv")
    print(f'Evaluation took {time.time() - eval_time} seconds')


