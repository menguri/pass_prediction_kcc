# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm

#패스 식별 방법
def identify_pass(row):
    #event-definition에 따라 정의된 모든 이벤트들 읽은 결과 총 패스의 유형들
    #row[type]이 PASS, BALL LOST, BALL OUT중 하나이고
    #row[subtype]이 해당 key의 value값중 하나이어야한다.
    pass_dict = {'PASS':['PASS','HEAD','CROSS','GOAL KICK','DEEP BALL','THROUGH BALL-DEEP BALL', 'CLEARANCE', 'HEAD-CLEARANCE',
                         'HEAD-INTERCEPTION-CLEARANCE', 'OFFSIDE','DEEP BALL-OFFSIDE'],
                 'BALL LOST' : ['INTERCEPTION', 'CROSS-INTERCEPTION','GOAL KICK-INTERCEPTION', 'CLEARANCE', 'GOAL KICK', 
                                'DEEP BALL', 'CLEARANCE-INTERCEPTION','THROUGH BALL-DEEP BALL-INTERCEPTION', 'CROSS'],
                 'BALL OUT' : ['CLEARANCE', 'CROSS', 'GOAL KICK', 'DEEP BALL', 'THROUGH BALL-DEEP BALL']
                 }

    if (row['type'] in pass_dict.keys()) and (row['subtype'] in pass_dict[row['type']]):
        return "Pass"
    else:
        return row["type"]

#패스에 대한 레이블링을 부여함(성공여부, 골여부, 자책골 여부)
def cleanse_metrica_event_data(df_events):
    """
    Function to clean the Metrica event data. Notice that quite a lot of the code is needed to make the Metrica data
    compatible with the Wyscout format
    :param game: (int) GameId
    :param reverse: (bool) If True, the away team is playing left to right in the first half
    :return: None
    """

    # identify goals and own goals
    df_events["goal"] = 1 * (
        df_events.apply(
            lambda row: row["type"] == "SHOT" and "-GOAL" in row["subtype"], axis=1
        )
    )
    df_events["ownGoal"] = 1 * (
        df_events.apply(
            lambda row: row["type"] == "BALL OUT" and "-GOAL" in row["subtype"], axis=1
        )
    )

    # set the teamId - home team is always 1, away team always 2
    df_events["teamId"] = np.where(df_events["team"] == "Home", 1, 2)

    ## identify passes
    #패스 성공 여부에 대한 레이블링 부여함
    df_events["eventName"] = df_events.apply(identify_pass, axis=1)

    return df_events

#soccermap형식으로 변환하기 위해서 하나의 컬럼(freeze_frame)에 선수(key)별 정보(value)방식으로 저장해놓음
#변환할 때, 코드를 단축해주고 시간을 단축해주기 때문에 사용
def make_freeze_frame(events, traces):
    events['freeze_frame'] = dict()

    for idx, row in tqdm(events.iterrows()):
        start_frame, end_frame = row['start_frame'], row['end_frame']
        frame_traces = traces[(traces['frame'] >= start_frame) & (traces['frame']  <= end_frame)]

        # EPV연산시 SoccerMapTensor형태로 빠르게 변환하기 위한 데이터 freeze_frame
        # tracking-data를 수집할 수 없는 이벤트는 제외
        # pass-data가 아닌 이벤트는 제외함(ball-drive, shot은 보류)
        if frame_traces.empty:
            events.at[idx,'freeze_frame'] = {}
            continue

        #frame에 해당하는 인덱스는 어차피 한개
        start_index = frame_traces.index[0]
        end_index = frame_traces.index[-1]

        freeze_dict = {}
        
        #이벤트를 수행하는 선수과 같은팀 선수 / 상대팀 선수를 분류
        passer = row['from']
        team = passer[0]

        teammates = [c[:-2] for c in frame_traces.dropna(axis=1).columns if c[0] == team and c.endswith("_x")]
        opoonents = [c[:-2] for c in frame_traces.dropna(axis=1).columns if c[0] != team and c[:4] != "ball" and c.endswith("_x")]

        for col in teammates:
            teammate = True
            actor = True if col == passer else False
            ball = False

            start_x = frame_traces.at[start_index, f'{col}_x']
            start_y = frame_traces.at[start_index, f'{col}_y']
            start_vx = frame_traces.at[start_index, f'{col}_vx']
            start_vy = frame_traces.at[start_index, f'{col}_vy']
            start_speed = frame_traces.at[start_index, f'{col}_speed']

            end_x = frame_traces.at[end_index, f'{col}_x']
            end_y = frame_traces.at[end_index, f'{col}_y']
            end_vx = frame_traces.at[end_index, f'{col}_vx']
            end_vy = frame_traces.at[end_index, f'{col}_vy']
            end_speed = frame_traces.at[end_index, f'{col}_speed']

            teammate_info_dict = {'teammate': teammate,'actor':actor,'ball':ball,'start_x':start_x,'start_y':start_y,'start_vx':start_vx,'start_vy':start_vy,
                                                                        'end_x':end_x,'end_y':end_y,'end_vx':end_vx,'end_vy':end_vy}

            freeze_dict[col] = teammate_info_dict   

        for col in opoonents:
            teammate = False
            actor = False
            ball = False

            start_x = frame_traces.at[start_index, f'{col}_x']
            start_y = frame_traces.at[start_index, f'{col}_y']
            start_vx = frame_traces.at[start_index, f'{col}_vx']
            start_vy = frame_traces.at[start_index, f'{col}_vy']
            start_speed = frame_traces.at[start_index, f'{col}_speed']

            end_x = frame_traces.at[end_index, f'{col}_x']
            end_y = frame_traces.at[end_index, f'{col}_y']
            end_vx = frame_traces.at[end_index, f'{col}_vx']
            end_vy = frame_traces.at[end_index, f'{col}_vy']
            end_speed = frame_traces.at[end_index, f'{col}_speed']

            opoonent_info_dict = {'teammate': teammate,'actor':actor,'ball':ball,'start_x':start_x,'start_y':start_y,'start_vx':start_vx,'start_vy':start_vy,
                                                                        'end_x':end_x,'end_y':end_y,'end_vx':end_vx,'end_vy':end_vy}
            freeze_dict[col] = opoonent_info_dict  

        for col in ["ball"]:
            teammate = False
            actor = False
            ball = True

            start_x = frame_traces.at[start_index, f'{col}_x']
            start_y = frame_traces.at[start_index, f'{col}_y']

            end_x = frame_traces.at[end_index, f'{col}_x']
            end_y = frame_traces.at[end_index, f'{col}_y']

            ball_info_dict = {'teammate': teammate,'actor':actor,'ball':ball,'start_x':start_x,'start_y':start_y,'start_vx':None,'start_vy':None,
                                                                        'end_x':end_x,'end_y':end_y,'end_vx':None,'end_vy':None}
            
            freeze_dict[col] = ball_info_dict   

        events.at[idx,'freeze_frame'] = freeze_dict

    return events

#항상 home팀이 왼쪽 & away팀이 오른쪽에 배치시키도록함
def rotate_pitch(reverse, events, field_dimen):
    session = 1 if reverse else 2
    events.loc[events['session'] == session,'freeze_frame'] = events.loc[events['session'] == session].apply(lambda row: freeze_left_to_right(row, field_dimen), axis=1)

    return events

def freeze_left_to_right(actions, field_dimen):
    freezedf = pd.DataFrame.from_records(actions["freeze_frame"])

    for player in freezedf.keys():
        freezedf[player]["start_x"] = field_dimen[0] - freezedf[player]["start_x"]
        freezedf[player]["start_y"] = field_dimen[1] - freezedf[player]["start_y"]     
        freezedf[player]["end_x"] = field_dimen[0] - freezedf[player]["end_x"]
        freezedf[player]["end_y"] = field_dimen[1] - freezedf[player]["end_y"]     

        if player != 'ball':
            freezedf[player]["start_vx"] = -freezedf[player]["start_vx"]
            freezedf[player]["start_vy"] = -freezedf[player]["start_vy"]       
            freezedf[player]["end_vx"] = -freezedf[player]["end_vx"]
            freezedf[player]["end_vy"] = -freezedf[player]["end_vy"]      

    return freezedf.to_dict()   

def preprocess_data(game_id, field_dimen=(106,68)):
    # 전반이든 후반이든 하상 home -> away로 단일방향으로 설정함(추후에 공격방향이 왼쪽에 배치시킬 때 유용하게 활용함)
    # metrica1 : 전반전 : home(A) - away(B)
    # 	         후반적 : away(B) - home(A)

    # metrica2 : 전반전 :  away(B) - hOME(A)
    # 	         후반적 :  home(A) - away(B)

    # metrica3 : 전반전 :  home(B) - away(A)
    # 	         후반적 :  away(A) - home(B)

    #metrica1, 3경기는 후반전을 flip하고, 2경기는 전반전을 flip함
    reverse_dict = {1:False, 2:True, 3:False}

    events = pd.read_csv(f'../data/preprocess-data/event-data/match{game_id}.csv')
    traces = pd.read_csv(f'../data/preprocess-data/tracking-data/match{game_id}.csv')

    events = cleanse_metrica_event_data(events)
    events = make_freeze_frame(events, traces)
    events = rotate_pitch(reverse_dict[game_id], events, field_dimen)
 
    return events, traces
