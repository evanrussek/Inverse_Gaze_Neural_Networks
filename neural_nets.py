import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# set up neural networks
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.lstm(x)[0]
        x = self.linear(h)
        return x
    
    def get_states_across_time(self, x):
        h_c = None
        h_list, c_list = list(), list()
        with torch.no_grad():
            for t in range(x.size(1)):
                h_c = self.lstm(x[:, [t], :], h_c)[1]
                h_list.append(h_c[0])
                c_list.append(h_c[1])
            h = torch.cat(h_list)
            c = torch.cat(c_list)
        return h, c
    
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=.2):
        super().__init__()

        self.input_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_output = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        ha = self.input_hidden(x)
        hb = F.relu(ha)
        o = self.hidden_output(hb)
        return o

# set up neural networks
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=.2):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.gru(x)[0]
        x = self.linear(h)
        return x

# 2 parameter model for the choice-only
class SimpleChoiceOnly(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.input_output = nn.Linear(1,1)

    def forward(self, x):
        
        
        o = torch.zeros(x.shape)
        for item in range(3): # ?
            
            a = torch.zeros(x.shape[0],1)
            a[:,0] = x[:,item]
            o[:,item] = torch.squeeze(self.input_output(a))

        return o

    
#### now create the transfoermer model 
# we want a embedding -> position encoding -> transformer encoder -> linear readout -- for now, just a single layer
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    # d_model? 
    # change dropout to 0? -- so it matches RNNs, --- was .1
    def __init__(self, d_model, dropout=0, max_len=300): # what if you change this to the max length? ? - what is the max length?
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
#        pe[:, 1::2] = torch.cos(position * div_term)
        # https://discuss.pytorch.org/t/transformer-example-position-encoding-function-works-only-for-even-d-model/100986/3
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        
        # print(x.shape)
        # print(self.pe.shape)
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

    
# now make a transformer - https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class SimpleTransformer(nn.Module):
    def __init__(self, n_token, d_model, dim_feedforward, output_size, nlayers = 1, nhead = 1, dropout=0): # do you want dropout? this was .5 before...
        super().__init__()
        
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
                        
        self.src_mask = None
        
        # encoder
        self.encoder = nn.Linear(n_token, d_model) # blow up to the hidden size
        
        # position encoder
        self.pos_encoder = PositionalEncoding(d_model, dropout = dropout)
        
        # Transformer (embedding -> multihead_attention -> 
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first = True) # nhead???
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = d_model
        self.decoder = nn.Linear(d_model, output_size)
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:
                mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
                self.src_mask = mask
            # if self.src_mask is None or self.src_mask.size(0) != len(src):
                # mask = self._generate_square_subsequent_mask(len(src)).to(device)
                # self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output#F.log_softmax(output, dim=-1)

# parameters - hidden size, learning rate, nheads?, nlayers?
