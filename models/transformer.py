# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
from .base import ModelBase
import math
from pathlib import Path

from .base import ModelBase

class TransformerModel(ModelBase):
    """
    Note that this class assumes feature_engineering was run with channels_first=True

    Parameters
    ----------
    in_channels: int, default=9
        Number of channels in the input data. Default taken from the number of bands in the
        MOD09A1 + the number of bands in the MYD11A2 datasets
    num_bins: int, default=32
        Number of bins in the histogram
    hidden_size: int, default=128
        The size of the hidden state. Default taken from the original repository
    dropout_rate: float, default=0.75
        Default taken from the original paper. Note that this dropout is applied to the
        hidden state after each timestep, not after each layer (since there is only one layer)
    dense_features: list, or None, default=None.
        output feature size of the Linear layers. If None, default values will be taken from the paper.
        The length of the list defines how many linear layers are used.
    savedir: pathlib Path, default=Path('data/models')
        The directory into which the models should be saved.
    device: torch.device
        Device to run model on. By default, checks for a GPU. If none exists, uses
        the CPU
    """

    def __init__(
        self,
        in_channels=9,
        num_bins=32,
        hidden_size=128,
        dropout_rate=0.75,
        num_heads=4,
        dense_features=None,
        num_encoders=1,
        savedir=Path("data/models"),
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.01,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):

        model = TransformerNet(
            in_channels=in_channels,
            num_bins=num_bins,
            num_encoders=num_encoders,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dense_features=dense_features
        )

        if dense_features is None:
            num_dense_layers = 2
        else:
            num_dense_layers = len(dense_features)
        model_weight = f"dense_layers.{num_dense_layers - 1}.weight"
        model_bias = f"dense_layers.{num_dense_layers - 1}.bias"

        super().__init__(
            model,
            model_weight,
            model_bias,
            "transformer",
            savedir,
            use_gp,
            sigma,
            r_loc,
            r_year,
            sigma_e,
            sigma_b,
            device,
        )

    def reinitialize_model(self, time=None):
        pass
        # self.model.initialize_weights()



class TransformerNet(nn.Module):
    def __init__(self, in_channels, num_bins, num_encoders, num_heads, dropout_rate, dense_features):
        super(TransformerNet, self).__init__()
        
        self.encoder = TransformerEncoder(
            embed_dim=in_channels*num_bins, 
            num_encoders=num_encoders, 
            dropout_rate=dropout_rate,
            dense_features=dense_features, 
            num_heads=num_heads
        )

    def forward(self, src):
        enc_out = self.encoder(src)
        return enc_out
    


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_encoders, dropout_rate, dense_features, num_heads):
        super(TransformerEncoder, self).__init__()
        
        #self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(32, embed_dim) # 32 = sequence_length

        self.layers = nn.ModuleList([TransformerBlock(embed_dim=embed_dim, dropout_rate=dropout_rate, dense_features=dense_features, n_heads=num_heads) for i in range(num_encoders)])
    
    def forward(self, x):
        #embed_out = self.embedding_layer(x)
        out = self.positional_encoder(x)
        for layer in self.layers:
            out = layer(out,out,out) # ?

        return out  #32x32x(32*9) = 32x32x288 = batch * seq_len * embed_dim (bins_size * num_bands)
    


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, dropout_rate, dense_features, n_heads):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, 4*embed_dim),
                          nn.ReLU(),
                          nn.Linear(4*embed_dim, 1)
        )

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self,key,query,value):   
        attention_out = self.attention(key,query,value)  
        attention_residual_out = attention_out + value  
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) 

        feed_fwd_out = self.feed_forward(norm1_out) 
        feed_fwd_residual_out = feed_fwd_out + norm1_out 
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out
    

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_model_dim, max_seq_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_seq_len, embed_model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_model_dim, 2).float() * (-math.log(10000.0) / embed_model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)
       
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim 
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        
        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.out(concat) #(32,10,512) -> (32,10,512)
       
        return output