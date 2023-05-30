from torch import nn
import torch
import numpy as np
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
        hidden_size=288,
        dropout_rate=0.75,
        num_heads=3,
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
            hidden_size=hidden_size,
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
        self.model.initialize_weights()


class TransformerNet(nn.Module):
    def __init__(
        self,
        in_channels=9,
        num_bins=32,
        hidden_size=2048, # not used currently
        num_encoders=5,
        num_heads=3,
        dropout_rate=0.75,
        dense_features=None
    ):

        super().__init__()
        
        if dense_features is None: # output of self-attention layer is gonna be: 64 x 32 x 288, so the first dense layer should have 288 input features (dimension not seq length)
            dense_features = [in_channels*num_bins, 1]
        dense_features.insert(0, in_channels*num_bins)

        self.dropout = nn.Dropout(dropout_rate)

        # Consider creating a linear layer to map the input size to the hidden size
        # self.input_layer = nn.Linear(in_channels*num_bins, hidden_size)

        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        # TransformerEncoderLayer is made up of self-attention and feedforward network
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels*num_bins, nhead=num_heads, dim_feedforward=2048, dropout=dropout_rate, batch_first=False)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.transformer = transformer_encoder

        self.hidden_size = in_channels*num_bins
        
        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=dense_features[i - 1], out_features=dense_features[i]
                )
                for i in range(1, len(dense_features))
            ]
        )

        self.initialize_weights()


    def forward(self, x, return_last_dense=False):
        # the model expects feature_engineer to have been run with channels_first=True, which means
        # the input is [batch, bands, times, bins].
        # Reshape to [times, batch, bands * bins] as needed for the transformer 
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[1], x.shape[0], x.shape[2] * x.shape[3])
        encoded = self.transformer(x) # [times, batch, bands * bins] = 32 * 32 * 288 (288 = 9 * 32) 
        # Reshape to # [batch, times, bands * bins] as needed for the dense layers: [32, 32, 288] or [64, 32, 288] if batch size = 64
        encoded = encoded.view(x.shape[1], x.shape[0], x.shape[2])
        # Average over the sequence dimension
        encoded = torch.mean(encoded, dim=1) # [batch, bands * bins]
        input_encoded = encoded
        for _, layer in enumerate(self.dense_layers):
            encoded = layer(encoded) # input features = bins*bands, output features = 1
            encoded = nn.ReLU()(encoded)
            """# minimum between the number of dense layers and the number of encoder layers - 1
            if return_last_dense and (layer_number == len(self.dense_layers) - 2):
                output = encoded # this line is never reached """
            
        if return_last_dense:
            return encoded, input_encoded
        return encoded
    

    def initialize_weights(self):
        """sqrt_k = math.sqrt(1 / self.hidden_size)
        for parameters in self.rnn.all_weights:
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)"""

        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.weight.data)
            nn.init.constant_(dense_layer.bias.data, 0)















    """def initialize_weights(self):
        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.weight.data)
            nn.init.constant_(dense_layer.bias.data, 0)"""
        
    # Cant access the weights of the transformer encoder layers, probabyl already initialized automatically
    """for parameters in self.transformer.all_weights:
            for pam in parameters:
                nn.init.uniform_(pam.data, -0.1, 0.1)"""
            
    
"""

class TransformerNet(nn.Module):
    def __init__(self, in_channels, num_bins, hidden_size, num_heads, num_encoders, dropout_rate):
        super(TransformerNet, self).__init__()
        # Fix this
        self.encoder_stack = nn.ModuleList([
            TransformerEncoder(in_channels*num_bins, hidden_size, num_heads, dropout_rate)
            for _ in range(num_encoders)
        ])
        self.hidden_size = hidden_size

        self.fc = nn.Linear(hidden_size, 1)

        self.initialize_weights()

    def initialize_weights(self):

        sqrt_k = math.sqrt(1 / self.hidden_size)

        for encoder in self.encoder_stack:
            for parameters in encoder.all_weights:
                for pam in parameters:
                    nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)

        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.weight.data)
            nn.init.constant_(dense_layer.bias.data, 0)

    def forward(self, x):
        for encoder in self.encoder_stack:
            x = encoder(x)

        # Aggregate the output of the last encoder layer
        x = torch.mean(x, dim=1) # ?
        x = self.fc(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # self.embedding = nn.Linear(input_dim, hidden_dim) # [TODO]: try to use embedding layer instead of the whole histograms vectors.
        self.position_encoding = PositionalEncoding(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.attention_head = MultiHeadAttention(input_dim, hidden_dim, num_heads) # input_dim
        self.feed_forward = FeedForward(hidden_dim)
        self.all_weights = [self.attention_head.all_weights, self.feed_forward.all_weights]
        
        

    def forward(self, x):
        # x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        x = self.attention_head(x)
        x = self.feed_forward(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_length=5000):
        super(PositionalEncoding, self).__init__()

        self.hidden_dim = hidden_dim

        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        self.position_encoding = torch.zeros((1, max_length, hidden_dim))

        self.position_encoding[0, :, 0::2] = torch.sin(position * div_term)
        self.position_encoding[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(1), :] # add skip connections
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim) # ?
        self.all_weights = [self.query.weight, self.key.weight, self.value.weight, self.fc.weight]

    def forward(self, x):
        batch_size = x.size(0)

        # Split the input into multi-heads
        queries = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context = torch.matmul(attention_probs, values)
        # Combine the num_heads and sequence_length dimensions into a single dimension
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # Apply skip connection and linear layer
        residual = self.layer_norm(x + context)
        output = self.fc(residual)

        return output


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(FeedForward, self).__init__()

        self.hidden_dim = hidden_dim

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )

        self.all_weights = [self.ff[0].weight, self.ff[2].weight]

    def forward(self, x):
        return self.ff(x)
"""