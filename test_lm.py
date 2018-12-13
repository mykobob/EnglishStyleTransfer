import argparse
import torch
import utils
import pickle

from LM_models import RNNLM

parser = argparse.ArgumentParser()
parser.add_argument('--model_path')
parser.add_argument('--indexer_path')
args = parser.parse_args()

# Load the best saved model.
with open(args.model_path, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

with open(args.indexer_path, 'rb') as f:
    indexer = pickle.load(f)

# Run on test data.
#test_loss = evaluate(test_data, test_data_lens)
#print('=' * 89)
#print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#    test_loss, math.exp(test_loss)))
#print('=' * 89)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
