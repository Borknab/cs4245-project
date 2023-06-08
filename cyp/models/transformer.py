# [CS4245]
from torch import nn
import torch
import numpy as np
import math
from pathlib import Path
import torch.nn.functional as F
import pandas as pd
from .base import ModelBase

class TransformerModel(ModelBase):
    def __init__(
        self,
        in_channels=9,
        num_bins=32,
        hidden_size=288,
        dropout_rate=0.5,
        num_heads=3,
        dense_features=None,
        num_encoders=5,
        savedir=Path("data/models"),
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.01,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        patience=10,
    ):

        model = TransformerNet(
            in_channels=in_channels,
            num_bins=num_bins,
            hidden_size=hidden_size,
            num_encoders=num_encoders,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dense_features=dense_features,
            device=device
        )

        if dense_features is None:
            num_dense_layers = 2
        else:
            num_dense_layers = len(dense_features)

        model_weight = f"dense_layers.{num_dense_layers - 1}.weight"
        model_bias = f"dense_layers.{num_dense_layers - 1}.bias"

        
        hyperparameters_df = pd.DataFrame(
            {
                "hidden_size": hidden_size,
                "dropout_rate": dropout_rate,
                "num_dense_layers": num_dense_layers,
                "num_encoders": num_encoders,
                "num_heads": num_heads,
                "patience": patience,
            },
            index=[0],
        )

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
            hyperparameters_df 
        )

    def reinitialize_model(self, time=None):
        self.model.initialize_weights()



class TransformerNet(nn.Module):
    def __init__(
        self,
        in_channels=9,
        num_bins=32,
        hidden_size=1024, # not used currently
        num_encoders=5,
        num_heads=3,
        dropout_rate=0.75,
        dense_features=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):

        super().__init__()
        
        if dense_features is None:
            dense_features = [in_channels*num_bins, 1]
        dense_features.insert(0, in_channels*num_bins)

        self.dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels*num_bins, nhead=num_heads, dim_feedforward=1024, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.hidden_size = in_channels*num_bins
        
        self.dense_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(
                        in_features=dense_features[i - 1], out_features=dense_features[i]
                    ),
                    nn.ReLU()
                )
                for i in range(1, len(dense_features) - 1)
            ]
        )
        # No activation after the last dense layer
        self.dense_layers.append(nn.Linear(dense_features[-2], dense_features[-1]))

        self.device = device
        
        self.initialize_weights()

        # Add an attention pooling layer
        self.attention_pool = AttentionPool(in_channels*num_bins)


    def forward(self, x, return_last_dense=False):
        x = x.to(self.device)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        positions = torch.arange(0, x.shape[1]).unsqueeze(1)
        pe = torch.zeros((1, x.shape[1], x.shape[2])).to(self.device)
        div_term = torch.exp(torch.arange(0, x.shape[2], 2) * -(math.log(10000.0) / x.shape[2]))
        pe[0, :, 0::2] = torch.sin(positions * div_term)
        pe[0, :, 1::2] = torch.cos(positions * div_term)
        x = x + pe

        encoded = self.transformer(x)

        # tried to do a global average pooling, but it didn't work well
        # encoded = encoded.mean(dim=1)
        # tried to do a global max pooling, but it didn't work well
        # encoded = encoded.max(dim=1)[0]

        # attention pool over time dimension
        encoded = self.attention_pool(encoded) 

        input_encoded = encoded
        for layer in self.dense_layers:
            encoded = layer(encoded)
        if return_last_dense:
            return encoded, input_encoded
        return encoded
    

    def initialize_weights(self):
        for dense_layer in self.dense_layers:
            if isinstance(dense_layer, nn.Linear):
                nn.init.kaiming_uniform_(dense_layer.weight.data)
                nn.init.constant_(dense_layer.bias.data, 0)


class AttentionPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, in_channels))

    def forward(self, x):
        # calculate attention scores
        att_scores = F.softmax((x * self.query).sum(-1), dim=-1)
        # apply attention scores
        output = (x * att_scores.unsqueeze(-1)).sum(1)
        return output