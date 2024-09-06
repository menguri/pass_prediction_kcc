import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


class IntendedReceiverDataset(Dataset):

    def __init__(self, PassDataSet: pd.DataFrame, n_features=6, flip_pitch=False, augment=False):
        self.n_features = n_features  # options: 2, 4, 6
        self.feature_types = ["_x", "_y", "_vx", "_vy", "_speed", "_accel"][:n_features]

        self.flip_pitch = flip_pitch
        self.augment = augment
        self.ps = (108, 72)

        passer_inputs = []
        teammate_inputs = []
        opponent_inputs = []
        ball_inputs = []

        team_poss_inputs = []
        event_player_inputs = []

        passer_mask_list = []
        receiver_labels_list = []
        receiver_loc_list = []

        # 학습에 사용하지 못해서 제외시킨 데이터 추출 : 테스트할 때 파악하기 위해
        self.except_index = []

        # 학습 시 선수 번호를 인코딩하여 사용하고 테스트 결과를 선수 번호로 다시 변환할 때, event-data와 test-data의 병합 과정에서 정렬 문제가 발생합니다
        game_ids = sorted(PassDataSet["game_id"].unique())

        # split으로 인해 test-data의 game_id순서가 불규칙해짐.
        traces_dict = {}
        for game_id in game_ids:
            traces = pd.read_csv(f"../data/preprocess-data/tracking-data/match{game_id}.csv")
            traces.set_index("frame", inplace=True)
            traces_dict[game_id] = traces

        for _, row in PassDataSet.iterrows():
            start_frame = row["start_frame"]
            game_id = row["game_id"]
            traces = traces_dict[game_id]

            margin = 10
            if start_frame + margin - 1 > row["end_frame"]:
                self.except_index.append(row["event_id"])
                continue

            session = traces.at[start_frame, "session"]
            frame_traces = traces.loc[start_frame - 60 : start_frame + margin - 1]
            frame_traces = frame_traces[frame_traces["session"] == session]

            passer = row["from"]
            team = passer[0]
            receiver = row["to"]

            # 6135가 왜 PASS로 이벤트를 부여했는지는 모르겠지만, passer=receiver같은 반례
            if passer == receiver:
                self.except_index.append(row["event_id"])
                continue

            # 선수 등번호를 기준으로 정렬
            # 각 선수 인덱스별 feature & label를 연관시키기 위한 방법
            # 교체 or 경기별로 player(label)가 변화기 때문에
            player_cols = [
                f"{c[:3]}{t}"
                for c in frame_traces.dropna(axis=1).columns
                if c.endswith("_speed") and not c.startswith("ball")
                for t in self.feature_types
            ]

            player_cols = sorted(
                player_cols,
                key=lambda c: (c[:3], self.feature_types.index(c[3:])),
            )

            player_names = [c[:3] for c in player_cols if c.endswith("_x") and c[0] == team]
            label_dict = {col: idx for idx, col in enumerate(player_names)}

            teammate_cols = [c for c in player_cols if c[0] == team]  # and c[:3] != passer]
            opponent_cols = [c for c in player_cols if c[0] != team]
            ball_cols = [c for c in frame_traces.dropna(axis=1).columns if c.startswith("ball")]

            stride = 1

            # 1개 간격(0.04초)으로 데이터 추출
            passer_input = frame_traces[[f"{passer}{t}" for t in self.feature_types]].values[::stride]
            teammate_input = frame_traces[teammate_cols].dropna(axis=1).values[::stride]
            opponent_input = frame_traces[opponent_cols].dropna(axis=1).values[::stride]
            ball_input = frame_traces[ball_cols].values[::stride]

            team_poss_values = frame_traces[["team_poss"]].values[::stride]
            event_player_values = frame_traces[["event_player"]].values[::stride]

            team_poss_input = np.zeros((len(team_poss_values), len(label_dict)))
            event_player_input = np.zeros((len(event_player_values), len(label_dict)))

            # team_poss 처리: team_poss에 해당하는 모든 선수 인덱스에 1 할당
            for t, team_poss in enumerate(team_poss_values):
                team_indices = dict(filter(lambda p: p[0][0] == team_poss, label_dict.items()))
                team_poss_input[t, list(team_indices.values())] = 1

            # event_player 처리: 각 시점의 event_player에 해당하는 인덱스에만 1 할당
            for t, event_player in enumerate(event_player_values):
                event_player_indice = dict(filter(lambda p: p[0] == event_player, label_dict.items()))
                event_player_input[t, list(event_player_indice.values())] = 1

            concat_input = np.concatenate(
                [
                    passer_input,
                    teammate_input,
                    opponent_input,
                    ball_input,
                    team_poss_input,
                    event_player_input,
                ],
                axis=1,
            )

            if (
                not np.isnan(concat_input).any()
                and teammate_input.shape[1] == n_features * 11
                and opponent_input.shape[1] == n_features * 11
            ):

                passer_inputs.append(torch.FloatTensor(passer_input))
                teammate_inputs.append(torch.FloatTensor(teammate_input))
                opponent_inputs.append(torch.FloatTensor(opponent_input))
                ball_inputs.append(torch.FloatTensor(ball_input))

                team_poss_inputs.append(torch.FloatTensor(team_poss_input))
                event_player_inputs.append(torch.FloatTensor(event_player_input))

                passer_mask_list.append(int(label_dict[passer]))
                receiver_labels_list.append(int(label_dict[receiver]))
                receiver_loc_list.append((row["end_x"], row["end_y"]))
            else:
                self.except_index.append(row["event_id"])

        # 각 경기별 각 움직임별 서로다른 움직임값들을 합치기 위한 작업
        # 학습데이터개수 X sequence X features
        self.passer_inputs = pad_sequence(passer_inputs).transpose(0, 1)
        self.teammate_inputs = pad_sequence(teammate_inputs).transpose(0, 1)
        self.opponent_inputs = pad_sequence(opponent_inputs).transpose(0, 1)
        self.ball_inputs = pad_sequence(ball_inputs).transpose(0, 1)

        self.team_poss_inputs = pad_sequence(team_poss_inputs).transpose(0, 1)
        self.event_player_inputs = pad_sequence(event_player_inputs).transpose(0, 1)

        self.passer_mask_list = torch.LongTensor(passer_mask_list)
        self.receiver_labels_list = torch.LongTensor(receiver_labels_list)
        self.receiver_loc_list = torch.FloatTensor(receiver_loc_list)

    def __getitem__(self, i):

        passer_input = self.passer_inputs[i].clone()
        teammate_input = self.teammate_inputs[i].clone()
        opponent_input = self.opponent_inputs[i].clone()
        ball_input = self.ball_inputs[i].clone()

        team_poss = self.team_poss_inputs[i].clone()
        event_player_input = self.event_player_inputs[i].clone()

        passer_mask_list = self.passer_mask_list[i].clone()
        receiver_labels_list = self.receiver_labels_list[i].clone()
        receiver_loc_list = self.receiver_loc_list[i].clone()

        # 좌-우 & 상-하 뒤집기
        if self.flip_pitch:
            flip_x = random.randint(0, 1)
            flip_y = random.randint(0, 1)

            ref_x = flip_x * self.ps[0]
            ref_y = flip_y * self.ps[1]
            mul_x = 1 - flip_x * 2
            mul_y = 1 - flip_y * 2

            passer_input[:, 0] = passer_input[:, 0] * mul_x + ref_x
            passer_input[:, 1] = passer_input[:, 1] * mul_y + ref_y

            ball_input[:, 0] = ball_input[:, 0] * mul_x + ref_x
            ball_input[:, 1] = ball_input[:, 1] * mul_y + ref_y

            teammate_input[:, 0 :: self.n_features] = teammate_input[:, 0 :: self.n_features] * mul_x + ref_x
            teammate_input[:, 1 :: self.n_features] = teammate_input[:, 1 :: self.n_features] * mul_y + ref_y

            opponent_input[:, 0 :: self.n_features] = opponent_input[:, 0 :: self.n_features] * mul_x + ref_x
            opponent_input[:, 1 :: self.n_features] = opponent_input[:, 1 :: self.n_features] * mul_y + ref_y

            if self.n_features > 2:
                passer_input[:, 2] = passer_input[:, 2] * mul_x
                passer_input[:, 3] = passer_input[:, 3] * mul_y

                teammate_input[:, 2 :: self.n_features] = teammate_input[:, 2 :: self.n_features] * mul_x
                teammate_input[:, 3 :: self.n_features] = teammate_input[:, 3 :: self.n_features] * mul_y

                opponent_input[:, 2 :: self.n_features] = opponent_input[:, 2 :: self.n_features] * mul_x
                opponent_input[:, 3 :: self.n_features] = opponent_input[:, 3 :: self.n_features] * mul_y

                ball_input[:, 2 :: self.n_features] = ball_input[:, 2 :: self.n_features] * mul_x
                ball_input[:, 3 :: self.n_features] = ball_input[:, 3 :: self.n_features] * mul_y

        if self.augment:
            random_shift_x = (random.random() - 0.5) * 50
            random_shift_y = (random.random() - 0.5) * 50

            passer_input[:, 0] += random_shift_x
            ball_input[:, 0] += random_shift_x
            teammate_input[:, 0 :: self.n_features] += random_shift_x
            opponent_input[:, 0 :: self.n_features] += random_shift_x

            passer_input[:, 1] += random_shift_y
            ball_input[:, 1] += random_shift_y
            teammate_input[:, 1 :: self.n_features] += random_shift_y
            opponent_input[:, 1 :: self.n_features] += random_shift_y

        return (
            passer_input,
            teammate_input,
            opponent_input,
            ball_input,
            team_poss,
            event_player_input,
            passer_mask_list,
            receiver_labels_list,
            receiver_loc_list,
        )

    # def __getitem__(self, i):
    #     return (
    #         self.passer_inputs[i],
    #         self.teammate_inputs[i],
    #         self.opponent_inputs[i],
    #         self.ball_inputs[i],

    #         self.team_poss_inputs[i],
    #         self.event_player_inputs[i],

    #         self.passer_mask_list[i],
    #         self.receiver_labels_list[i],
    #         self.receiver_loc_list[i],
    #     )

    def __len__(self):
        return len(self.receiver_labels_list)
