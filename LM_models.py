import torch.nn as nn
import torch.nn.functional as F

class FFNNLM(nn.Module):
    
    def __init__(self, window_size, hidden_size, embed_size, indexer):
        super(FFNNLM, self).__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.vocab = indexer
        self.hidden = nn.Linear(embed_size * window_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, len(self.vocab))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, cat_embeddings):
        hidden_out = self.relu(self.hidden(cat_embeddings))
        return self.softmax(self.classifier(hidden_out))

class RNNLM(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, bidirectional, dropout=0.5, tie_weights=False):
        super(RNNLM, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.GRU(input_size=ninp, hidden_size=nhid, num_layers=nlayers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.decoder = nn.Linear(nhid * (2 if bidirectional else 1), ntoken)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden = None

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, input_lens, hidden):
        emb = self.encoder(input)
        packed_embedding = nn.utils.rnn.pack_padded_sequence(emb, input_lens, batch_first=True)
        output, hidden = self.rnn(packed_embedding, hidden)
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        batch_size = output.size(0)
        output = output.contiguous().view(batch_size, -1, self.num_directions, self.nhid)
        
        classifier_input = output.view(output.size(0)*output.size(1), output.size(2) * output.size(3))
        decoded = self.decoder(classifier_input)
        decoded_reshaped = decoded.view(output.size(0), output.size(1), decoded.size(1))
        predictions = F.log_softmax(decoded_reshaped, dim=2)
        self.hidden = hidden
        return predictions, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers * self.num_directions, bsz, self.nhid)
