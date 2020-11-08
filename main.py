# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import logging
import data
from models.Transformer import TransformerModel
from models.LSTM import LSTMModel
from models.AttentionLSTM import AttentionLSTMLanguageModel


parser = argparse.ArgumentParser(description='PyTorch LSTM/ATT_LSTM/transformer Language Model')
parser.add_argument('--e', type=str, default=None,
                    help='name of the experiment')
parser.add_argument('--dataset', type=str, default='wikitext2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (lstm, att_lstm, transformer)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=60,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--patience', type=int, default=5,
                    help='number of bad epochs before stop the model')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--bias_reg', action='store_true',
                    help='when to use bias regulator on encoder')
parser.add_argument('--bias_reg_lambda', type=float, default=1.0,
                    help='bias regularization encoder loss weight factor')
parser.add_argument('--bias_reg_var_ratio', type=float, default=0.5,
                    help=('ratio of variance used for determining size of gender'
                          'subspace for bias regularization'))

args = parser.parse_args()
##############################################################################
# set initial configurations
##############################################################################
if args.e != None:
    SAVE_PATH = f'./checkpoints/{args.e}'
elif args.bias_reg:
    SAVE_PATH = f'./checkpoints/{args.model}/{args.dataset}/bias_reg_{args.bias_reg_lambda}'
else:
    SAVE_PATH = f'./checkpoints/{args.model}/{args.dataset}'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

logging.basicConfig(filename=f'{SAVE_PATH}/log_file.log', filemode='w', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(f'./data/{args.dataset}')

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

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.batch_size)
test_data = batchify(corpus.test, args.batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'transformer':
    model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
elif args.model == 'att_lstm':
    model = AttentionLSTMLanguageModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
else: 
    model = LSTMModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()

##############################################################################
# set bias reg needs
##############################################################################
filename  = f'./gender_pairs/{args.dataset}-gender-pairs'

female_words, male_words =[],[]
with open(filename,'r', encoding='utf8') as f:
    gender_pairs = f.readlines()

for gp in gender_pairs:
    m,f=gp.split()
    female_words.append(f)
    male_words.append(m)

gender_words = set(female_words) | set(male_words)

word2idx = corpus.dictionary.word2idx
#Gender pair indexes
D = torch.LongTensor([[word2idx[wf], word2idx[wm]]
                       for wf, wm in zip(female_words, male_words)
                       if wf in word2idx and wm in word2idx])

#Not gendered word indexes
N = torch.LongTensor([idx for w, idx in word2idx.items() if w not in gender_words])

def bias_regularization_encoder(model, D, N, var_ratio, lmbda, norm=True):
    """
    Compute bias regularization loss term
    Original code:  https://github.com/BordiaS/language-model-bias
    Original paper: https://arxiv.org/abs/1904.03035
    """
    W = model.encoder.weight
    if norm:
        W = W / model.encoder.weight.norm(2, dim=1).view(-1, 1)
    C = []
    # Stack all of the differences between the gender pairs
    for idx in range(D.size()[0]):
        idxs = D[idx].view(-1)
        u = W[idxs[0],:]
        v = W[idxs[1],:]
        C.append(((u - v)/2).view(1, -1))
    C = torch.cat(C, dim=0)
    # Get prinipal components
    U, S, V = torch.svd(C)

    # Find k such that we capture 100*var_ratio% of the gender variance
    var = S**2

    norm_var = var/var.sum()
    cumul_norm_var = torch.cumsum(norm_var, dim=0)
    _, k_idx = cumul_norm_var[cumul_norm_var >= var_ratio].min(dim=0)
    # Get first k components to for gender subspace
    
    B = V[:, :k_idx.data.item()+1]
    loss = torch.matmul(W[N], B).norm(2) ** 2

    return lmbda * loss


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

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


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


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        elif args.model == 'att_lstm':
            output, attention_scores = model(data)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            
        if args.bias_reg:
            bias_loss = bias_regularization_encoder(model, D, N, args.bias_reg_var_ratio, args.bias_reg_lambda, False)
            raw_loss = criterion(output, targets.view(-1))
            loss = raw_loss + bias_loss
        else:    
            loss = criterion(output, targets.view(-1))
        loss.backward()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break
        if bad_epochs == args.patience:
            break
# Loop over epochs.
lr = args.lr
best_val_loss = None
bad_epochs = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        logging.info('-' * 89)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | bad epochs:'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss), bad_epochs))
        logging.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(f'{SAVE_PATH}/best-model.pt', 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

        #early stop
        if round(val_loss,3) > round(best_val_loss,3):
            bad_epochs += 1

except KeyboardInterrupt:
    logging.info('-' * 89)
    logging.info('Exiting from training early')

# Load the best saved model.
model = torch.load(f'{SAVE_PATH}/best-model.pt')
# after load the rnn params are not a continuous chunk of memory
# this makes them a continuous chunk, and will speed up forward pass
# Currently, only rnn model supports flatten_parameters function.
if args.model in ['lstm', 'att_lstm']:
    model.lstm.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
logging.info('=' * 89)
logging.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging.info('=' * 89)