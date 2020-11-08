import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

# Based in: https://github.com/gautierdag/pytorch-attentive-lm
# Paper: https://www.aclweb.org/anthology/I17-1045/


class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()

        self.attn_1 = nn.Linear(feature_dim, feature_dim)
        self.attn_2 = nn.Linear(feature_dim, 1)

        #initialize weights
        nn.init.xavier_normal_(self.attn_1.weight)
        nn.init.xavier_normal_(self.attn_2.weight)

        #initialize biases
        self.attn_1.bias.data.fill_(0.0)
        self.attn_2.bias.data.fill_(0.0)

    def forward(self, x):
        sequence_length = x.shape[1]

        self_attention_scores = self.attn_2(torch.tanh(self.attn_1(x)))

        # Attend for each time step using the previous context
        context_vectors = []
        attention_vectors = []

        for t in range(sequence_length):
            weighted_attention_scores = F.softmax(self_attention_scores[:, : t + 1, :].clone(), dim=1)
            context_vectors.append(torch.sum(weighted_attention_scores * x[:, :t + 1, :], dim=1))

            attention_vectors.append(weighted_attention_scores.cpu().detach().numpy())

        context_vectors = torch.stack(context_vectors).transpose(0,1)

        return context_vectors, attention_vectors

    
class AttentionLSTMLanguageModel(nn.Module):
    def __init__(self, ntoken, 
                ninp = 300, 
                nhid = 65,
                nlayers = 1,
                dropout = 0.5,
                tie_weights = True ):
        super(AttentionLSTMLanguageModel, self).__init__()

        self.model_type = 'att_lstm'
        self.ninp, self.nhid, self.nlayers, self.ntoken = ninp, nhid, nlayers, ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(self.ntoken, self.ninp)
        self.lstm = nn.LSTM(self.ninp, self.nhid, 
                                self.nlayers, batch_first=True, 
                                dropout = dropout)

        self.attention_score_module = Attention(nhid)

        #layer to concatenate attention and hidden state
        self.concatenation_layer = nn.Linear(nhid * 2, nhid)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if self.ninp != nhid:
                raise ValueError('When using tied flag, encoder ninp must be equal to nhid')
            self.decoder.weight = self.encoder.weight
        
        self.init_weights()

    def forward(self, src):
        embedded = self.drop(self.encoder(src))

        self.flatten_parameters()
        
        output, _ = self.lstm(embedded)

        context_vectors, attention_score = self.attention_score_module(output)
        combined_encoding = torch.cat((context_vectors, output), dim=2)
        
        # concatenation layer
        output = torch.tanh(self.concatenation_layer(combined_encoding))
        output = self.drop(output)
        decoded = self.decoder(output.contiguous())

        return decoded.view(-1, self.ntoken), attention_score
    
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
            
    def flatten_parameters(self):
        self.lstm.flatten_parameters()

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
