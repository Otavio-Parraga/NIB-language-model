import torch
import argparse
import math
from models.LSTM import LSTMModel
from models.AttentionLSTM import AttentionLSTMLanguageModel
from models.Transformer import TransformerModel
import torch.nn as nn
import data

parser = argparse.ArgumentParser()

parser.add_argument('--split', type=str, default='valid')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--dataset', type=str, default='blogset',
                    help= 'blogset, br_lit or wikitext2')
parser.add_argument('--model', type=str, default='lstm',
                    help= 'model to be used')
parser.add_argument('--bptt', type=int, default=35,
                    help='bptt length')
parser.add_argument('--bias_reg', type=str, default=None)

args = parser.parse_args()
args.tied = True

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corpus = data.Corpus(f'./data/{args.dataset}')
ntokens = len(corpus.dictionary)

word2idx = corpus.dictionary.word2idx

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

val_data = batchify(corpus.valid, args.batch_size)
test_data = batchify(corpus.test, args.batch_size)

criterion = nn.CrossEntropyLoss()

MODEL_DIR = f'{args.model}/{args.dataset}/best-model.pt' if args.bias_reg == None else f'{args.model}/{args.dataset}/bias_reg_{args.bias_reg}/best-model.pt' 

model = torch.load(f'checkpoints/{MODEL_DIR}').to(device)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'transformer':
        hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            elif args.model == 'att_lstm':
                output, att_score = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

final_loss = evaluate(val_data if args.split == 'valid' else test_data)

print(f'Loss: {final_loss} | Perplexity: {math.exp(final_loss)}')