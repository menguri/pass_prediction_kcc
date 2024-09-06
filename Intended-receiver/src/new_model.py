import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from set_transformer.model import SetTransformer
from src.utils import get_params_str, parse_model_params


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class DualTransformer(nn.Module):
    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = [
            "n_features",
            "n_classes",
            "context_dim",
            "trans_dim",
            "dropout",
        ]
        self.params = parse_model_params(self.model_args, params, parser)
        self.params_str = get_params_str(self.model_args, params)

        self.n_features = params["n_features"]
        self.n_classes = params["n_classes"]
        self.context_dim = params["context_dim"]
        self.trans_dim = params["trans_dim"]
        self.dropout = params["dropout"] if "dropout" in params else 0

        self.passer_fc = nn.Linear(self.n_features, self.context_dim)
        self.teammate_st = SetTransformer(self.n_features, self.context_dim * 4)
        self.opponent_st = SetTransformer(self.n_features, self.context_dim * 4)
        self.ball_fc = nn.Linear(2, self.context_dim)
        self.whole_st = SetTransformer(
            self.n_features + 2, self.context_dim, embed_type="equivariant"
        )

        self.team_poss_st = SetTransformer(self.n_classes, self.context_dim)
        self.event_player_st = SetTransformer(self.n_classes, self.context_dim)
        embed_output_dim = self.context_dim * 4
        self.context_fc = nn.Sequential(
            nn.Linear(embed_output_dim, self.trans_dim), nn.ReLU()
        )

        self.time_pos_encoder = PositionalEncoding(self.trans_dim)
        self.time_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                self.trans_dim, 4, self.trans_dim * 2, self.dropout
            ),
            2,
        )

        self.player_pos_encoder = PositionalEncoding(self.trans_dim)
        self.player_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                self.trans_dim, 4, self.trans_dim * 2, self.dropout
            ),
            2,
        )

        self.input_fc = nn.Linear(self.n_features, self.trans_dim)

        self.output_fc = nn.Sequential(nn.Linear(self.trans_dim, 1))

    def forward(
        self,
        passer_input: torch.Tensor,
        teammate_input: torch.Tensor,
        opponent_input: torch.Tensor,
        ball_input: torch.Tensor,
        team_poss_input: torch.Tensor,
        event_player_input: torch.Tensor,
    ) -> torch.Tensor:

        passer_input = passer_input.transpose(0, 1)  # [seq_len, bs, n_features]
        teammate_input = teammate_input.transpose(
            0, 1
        )  # [seq_len, bs, n_features * n_teammates]
        opponent_input = opponent_input.transpose(
            0, 1
        )  # [seq_len, bs, n_features * n_opponents]
        ball_input = ball_input.transpose(0, 1).repeat(1, 1, 3)  # [seq_len, bs, 2]

        team_poss_input = team_poss_input.transpose(0, 1)
        event_player_input = event_player_input.transpose(0, 1)

        seq_len = passer_input.size(0)
        batch_size = passer_input.size(1)

        players_with_ball = torch.cat(
            [teammate_input, opponent_input, passer_input, ball_input], dim=2
        )

        time_z = self.time_pos_encoder(
            self.input_fc(
                players_with_ball.reshape(
                    seq_len, batch_size * (2 * self.n_classes + 2), -1
                )
            )
            # * math.sqrt(self.params["trans_dim"])
        )
        time_h = self.time_encoder(time_z)

        time_h_reshaped = time_h.reshape(
            seq_len, batch_size, (2 * self.n_classes + 2), -1
        )[-1].transpose(
            0, 1
        )  # Select the last time step and transpose

        # player_z = self.player_pos_encoder(
        #    time_h_reshaped  # * math.sqrt(self.params["trans_dim"])
        # )
        player_h = self.player_encoder(time_h_reshaped)  # [n_classes, batch, trans_dim]

        player_h_reshaped = player_h[: self.n_classes].transpose(
            0, 1
        )  # [batch, n_classes, trans_dim]

        return self.output_fc(player_h_reshaped).reshape(batch_size, self.n_classes)
