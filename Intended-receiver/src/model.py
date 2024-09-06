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


class Model(nn.Module):
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
        # self.team_poss_fc = nn.Sequential(nn.Linear(22, context_dim), nn.ReLU(), nn.Dropout(dropout))
        # self.event_player_fc = nn.Sequential(nn.Linear(22, context_dim), nn.ReLU(), nn.Dropout(dropout))

        # passer, ball, team_poss, event_player : 1
        # teammate : 4
        # opponent : 4
        embed_output_dim = self.context_dim * 4
        self.context_fc = nn.Sequential(
            nn.Linear(embed_output_dim, self.trans_dim), nn.ReLU()
        )

        if params["model"] == "transformer":
            self.pos_encoder = PositionalEncoding(self.trans_dim)
            self.trans_encoder = TransformerEncoder(
                TransformerEncoderLayer(
                    self.trans_dim, 4, self.trans_dim * 2, self.dropout
                ),
                2,
            )
        elif params["model"] == "lstm":
            self.rnn = nn.LSTM(
                self.trans_dim,
                self.trans_dim // 2,
                num_layers=2,
                dropout=dropout,
                bidirectional=True,
            )
        elif params["model"] == "gru":
            self.gru = nn.GRU(
                self.trans_dim,
                self.trans_dim // 2,
                num_layers=2,
                dropout=dropout,
                bidirectional=True,
            )
        else:
            print("model error : ", params["model"])
            exit()

        self.new_rnn = nn.LSTM(
            self.n_features,
            self.trans_dim,
            num_layers=2,
            dropout=self.dropout,
            bidirectional=True,
        )

        self.output_first_fc = nn.Linear(self.trans_dim * 2, 1)
        self.output_fc = nn.Sequential(nn.Linear(self.trans_dim, self.n_classes))

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
        ball_input = ball_input.transpose(0, 1)  # [seq_len, bs, 2]

        team_poss_input = team_poss_input.transpose(0, 1)
        event_player_input = event_player_input.transpose(0, 1)

        seq_len = passer_input.size(0)
        batch_size = passer_input.size(1)
        n_features = passer_input.size(2)
        n_teammates = teammate_input.size(2) // n_features
        n_opponents = opponent_input.size(2) // n_features

        h, _ = self.new_rnn(
            teammate_input.reshape(seq_len, batch_size * n_teammates, n_features)
        )

        return self.output_first_fc(h[-1]).reshape(batch_size, n_teammates)

        passer_z = self.passer_fc(passer_input.reshape(seq_len * batch_size, -1))
        teammate_z = self.teammate_st(
            teammate_input.reshape(seq_len * batch_size, n_teammates, -1)
        )
        opponent_z = self.opponent_st(
            opponent_input.reshape(seq_len * batch_size, n_opponents, -1)
        )
        ball_z = self.ball_fc(ball_input.reshape(seq_len * batch_size, -1))

        team_poss_z = self.team_poss_st(
            team_poss_input.reshape(seq_len * batch_size, 1, -1)
        )
        event_player_z = self.event_player_st(
            event_player_input.reshape(seq_len * batch_size, 1, -1)
        )

        # 교수님 코드
        passer_x = passer_input.reshape(seq_len * batch_size, 1, -1)
        teammate_x = teammate_input.reshape(seq_len * batch_size, n_teammates, -1)
        opponent_x = opponent_input.reshape(seq_len * batch_size, n_opponents, -1)
        ball_x = ball_input.reshape(seq_len * batch_size, 1, -1)

        teammate_ball_dist = teammate_x[:, :, :2] - ball_x
        opponent_ball_dist = opponent_x[:, :, :2] - ball_x
        passer_ball_dist = passer_x[:, :, :2] - ball_x

        passer_x = torch.cat([passer_x, passer_ball_dist], -1)
        teammate_x = torch.cat([teammate_x, teammate_ball_dist], -1)
        opponent_x = torch.cat([opponent_x, opponent_ball_dist], -1)

        whole_a_z = self.whole_st(torch.cat([passer_x, teammate_x], -2))[:, 0]
        whole_b_z = self.whole_st(torch.cat([passer_x, opponent_x], -2))[:, 0]
        whole_c_z = self.whole_st(torch.cat([passer_x, teammate_x, opponent_x], -2))[
            :, 0
        ]

        # print("passer_z : ",passer_z.shape)
        # print("teammate_z : ",teammate_z.shape)
        # print("opponent_z : ",opponent_z.shape)
        # print("ball_z : ",ball_z.shape)
        # print("team_poss_z : ",team_poss_z.shape)
        # print("event_player_z : ",event_player_z.shape)

        # z = torch.cat([passer_z, teammate_z, opponent_z, ball_z, team_poss_z, event_player_z], -1)
        # z = torch.cat([passer_z, ball_z, whole_a_z, whole_b_z, team_poss_z, event_player_z], -1)
        z = torch.cat([ball_z, whole_c_z, team_poss_z, event_player_z], -1)
        z = self.context_fc(z)  # [time * bs, trans]

        if self.params["model"] == "transformer":
            z = self.pos_encoder(
                z.reshape(seq_len, batch_size, -1) * math.sqrt(self.params["trans_dim"])
            )
            h = self.trans_encoder(z)  # [time, bs, trans]
            h = self.bi_gru(z.reshape(seq_len, batch_size, -1))
        elif self.params["model"] == "lstm":
            self.rnn.flatten_parameters()
            h, _ = self.rnn(z.reshape(seq_len, batch_size, -1))  # [time, bs, trans]
        elif self.params["model"] == "gru":
            self.gru.flatten_parameters()
            h, _ = self.gru(z.reshape(seq_len, batch_size, -1))
        else:
            print("model error : ", self.params["model"])
            exit()

        return self.output_fc(h[-1])  # [bs, class]
