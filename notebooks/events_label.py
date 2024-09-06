import os, sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

from src.features import intended_receiver, extract_player_pos, extract_pass
from src.visualization import plot_action
from src.preprocess_data import make_freeze_frame
from src.labels import get_intended_receiver


# dribble labeling
def dribble_label(events):

    carray_event = events[events['eventName']=='CARRY'][['eventName', 'start_frame', 'end_frame', 'from', 'event_id', 'game_id']].reset_index(drop=True)

    accurate = []
    for idx, row in events.iterrows():
        #event로 알아내기
        df_carry = events[(events['start_frame'] >= row['start_frame'])
                                 &(events['start_frame'] <= row['end_frame'])
                                 &(events['game_id'] == row['game_id'])]
        event_list = list(df_carry['eventName'].values)
        event_list_num = [1 for event in event_list if event in ["BALL LOST", "BALL OUT"]]
        if sum(event_list_num) >= 1:
                accurate.append("fail")
        else:
                accurate.append("success")

    events['result'] = accurate
    return events


# Shot labeling
def shot_label(events):
    shot_event = events[events['eventName']=='SHOT']
    shot_event['result'] = shot_event['goal']
    shot_event['result'] =  shot_event['result'].replace({0:'fail', 1:'success'})
    return shot_event


# Cross labeling
def cross_label(events):
    cross_event = events[events['eventName']=='CROSS']
    cross_event['result'] = cross_event['accurate']
    cross_event['result'] =  cross_event['result'].replace({0:'fail', 1:'success'})
    cross_event.head()
    return cross_event


# Pass labeling
def pass_label(events):
    # get all passes
    pass_event = events[(events["eventName"] == "Pass")
                            & (events["start_frame"] < events["end_frame"])].copy()
    pass_event['result'] = pass_event['accurate']
    pass_event['result'] =  pass_event['result'].replace({0:'fail', 1:'success'})
    return pass_event


# events Concat
def all_label(events):
    # CROSS를 PASS와 분리
    events.loc[events['subtype'].str.contains('CROSS'), 'eventName'] = 'CROSS'
    final_events = events.copy()

    pass_event = pass_label(events)
    carry_event = dribble_label(events)
    shot_event = shot_label(events)
    cross_event = cross_label(events)

    pass_carry_shot = pd.concat([pass_event, carry_event, shot_event, cross_event], axis=0)
    final_events = pd.merge(final_events, pass_carry_shot[['event_id', 'result']], how='left', on='event_id')
    return final_events

