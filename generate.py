import torch
from models.LSTM import LSTMModel
from models.AttentionLSTM import AttentionLSTMLanguageModel
from models.Transformer import TransformerModel
import data
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--word', type=str, default=None)
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--dataset', type=str, default='blogset')
parser.add_argument('--temperature', type=float, default=1.)
parser.add_argument('--limit', type=int, default=1000)
parser.add_argument('--bias_reg', type=str, default=None)
parser.add_argument('--number', type=int, default=1)

args = parser.parse_args()
args.tied = True

FILE_NAME = f'{args.model}_{args.dataset}_{args.temperature}_{args.limit}.txt' if args.bias_reg == None else f'{args.model}_{args.dataset}_{args.temperature}_{args.limit}_bias_reg_{args.bias_reg}.txt' 
MODEL_DIR = f'{args.model}/{args.dataset}/best-model.pt' if args.bias_reg ==None else f'{args.model}/{args.dataset}/bias_reg_{args.bias_reg}/best-model.pt' 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corpus = data.Corpus(f'./data/{args.dataset}')

model = torch.load(f'checkpoints/{MODEL_DIR}').to(device)

model_type =  model.model_type if hasattr(model, 'model_type') else 'lstm'

if model_type != 'transformer':
    hidden = model.init_hidden(1)

model.eval()

idx2word = corpus.dictionary.idx2word
word2idx = corpus.dictionary.word2idx
ntokens = len(corpus.dictionary)

print(f'Runs: {args.number} | Model: {args.model} | Dataset: {args.dataset} | Bias_reg: {args.bias_reg} | Temperature: {args.temperature}')
for j in tqdm(range(args.number)):
    if args.word != None:
        input_tensor = torch.tensor([[word2idx[args.word]]]).to(device)
        
    else:
        input_tensor = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    final_string = f'{idx2word[input_tensor.item()]} '

    i = 0

    with torch.no_grad():
        with open(f'generated_text/{FILE_NAME}', 'a+', encoding='utf-8') as f:
            while True:
                if model_type == 'transformer':
                    output = model(input_tensor)

                elif model_type == 'att_lstm':
                    output, _ = model(input_tensor)

                elif model_type == 'lstm':
                    output, hidden = model(input_tensor, hidden)

                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input_tensor.fill_(word_idx)
                word = idx2word[word_idx]

                if i < args.limit:
                    final_string = final_string + word + ' '
                else:
                    f.write(final_string + '\n\n')
                    #print(final_string)
                    break
                i += 1