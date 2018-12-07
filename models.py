import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0" if False and torch.cuda.is_available() else "cpu")

# Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
# Works for both non-batched and batched inputs
class EmbeddingLayer(nn.Module):
    # Parameters: dimension of the word embeddings, number of words, and the dropout rate to apply
    # (0.2 is often a reasonable value)
    def __init__(self, input_dim, full_dict_size, embedding_dropout_rate):
        # print('input_dim', input_dim, 'full_dict_size', full_dict_size, 'embed_dropout', embedding_dropout_rate)
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    # Takes either a non-batched input [sent len x input_dim] or a batched input
    # [batch size x sent len x input dim]
    def forward(self, input):
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


# One-layer RNN encoder for batched inputs -- handles multiple sentences at once. You're free to call it with a
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class RNNEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, dropout, bidirect):
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=dropout, bidirectional=self.bidirect)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    # embedded_words should be a [batch size x sent len x input dim] tensor
    # input_lens is a tensor containing the length of each input sentence
    # Returns output (each word's representation), context_mask (a mask of 0s and 1s
    # reflecting where the model's output should be considered), and h_t, a *tuple* containing
    # the final states h and c from the encoder for each sentence.
    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        max_length = input_lens.data[0].item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            forward, backward = hn[0], hn[1]
#             print(hn[0].shape, hn[1].shape, hn.shape, type(hn))
            # Grab the representations from forward and backward LSTMs
            h_ = torch.cat((forward, backward), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            h_t = new_h
        else:
            h = hn[0][0]
            h_t = (h,)
        return (output, context_mask, h_t)

class RNNDecoder(nn.Module):

    def __init__(self, input_size, encoder_hidden_size, hidden_size, output_size, dropout, attention=True):
        super(RNNDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.attention_hidden_size = encoder_hidden_size + hidden_size
        self.attention = attention
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True,
                           dropout=dropout)
        self.hidden2vocab = nn.Linear(self.attention_hidden_size, output_size, bias=True)
        # self.hidden2vocab = nn.Linear(self.hidden_size, output_size, bias=True)

        self.attention_mat = nn.Bilinear(hidden_size, encoder_hidden_size, 1)

        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.hidden2vocab.weight)
        nn.init.xavier_uniform_(self.attention_mat.weight)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

    def get_output_size(self):
        return self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    def attention_function(self, dec_output, src_hidden):
        return self.attention_mat(dec_output, src_hidden)

    def calculate_weights(self, dec_output, src_sentence_hidden):
        num_src_words = src_sentence_hidden.size()[0]
        dup_dec_output = torch.cat([dec_output] * num_src_words)
        weights = self.attention_function(dup_dec_output, src_sentence_hidden)
        return torch.t(weights)

    def word_embed_weights(self, dec_output, src_sentence_hidden):
        weights = self.calculate_weights(dec_output, src_sentence_hidden)
        normalized_weights = F.softmax(weights, dim=1)
        return normalized_weights

    def get_attention(self, dec_output, src_sentence_hidden):
        word_weights = self.word_embed_weights(dec_output, src_sentence_hidden)
        return torch.mm(word_weights, src_sentence_hidden)

    def forward(self, input_hidden, embedded_word, src_sentence_embed):
        output, self.hidden = self.rnn(embedded_word, input_hidden)
        output = output.squeeze(0)

        if self.attention:
            src_sentence_embed = src_sentence_embed.squeeze(1)
            attention = self.get_attention(output, src_sentence_embed)
            hidden_layer_input = torch.cat((attention, output), dim=1)
        else:
            hidden_layer_input = output

        vocab_output = self.hidden2vocab(hidden_layer_input)
        vocab_prob = F.log_softmax(vocab_output, dim=1).squeeze(0)
        return vocab_prob

class VAE(nn.Module):
    
    def __init__(self, input_size, encoder_hidden_size, bottleneck_size, decoder_hidden_size, output_size, dropout, bidirect):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.encoder = RNNEncoder(input_size, encoder_hidden_size, dropout, bidirect)
        self.bottleneck_mu = self.Linear(encoder_hidden_size, bottleneck_size)
        self.bottleneck_sigma = self.Linear(encoder_hidden_size, bottleneck_size)
        self.decoder = RNNDecoder(bottleneck_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout, attention=False)
    
    def init_weight(self):
        self.encoder.init_weight()
        nn.init.xavier_uniform_(self.bottleneck_mu)
        nn.init.xavier_uniform_(self.bottleneck_sigma)
        self.decoder.init_weight()

    def encode(self, embedded_words, input_lens):
        output, mask, hidden = self.encoder(embedded_words, input_lens)
        return self.bottleneck_mu(output), self.bottleneck_sigma(output)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

