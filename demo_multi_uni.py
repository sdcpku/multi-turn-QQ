#! -*- coding:utf-8 -*-

# 使用tensorflow rc0.11时需加上下面两行代码
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import model_from_json
from openpyxl import Workbook
from data_auxiliary import fetch_chats
import numpy as np
from collections import deque
from calculate_reward import sent_embedding
from dqn_cnn_rnn import update_state

def choose_action_mul(state, actions, dqn):
    state_vec = np.concatenate(np.array([sent_embedding(sent) for sent in state]), axis=1)
    print state_vec.shape
    actions_vec = [np.array(sent_embedding(action)) for action in actions]
    print actions_vec[0].shape
    qvals = [dqn.predict([state_vec, action_vec]) for action_vec in actions_vec]
    act_num = np.argmax(qvals)
    return (actions[act_num], qvals[act_num])

def choose_action_uni(candi_QAs):
    actions = candi_QAs.keys()
    scores = np.array([candi_QAs[action][0] for action in actions])
    # print scores
    ind = (scores >= 80)
    return (np.array(actions)[ind], scores[ind])

def demo_multi_uni(chats):
    # 建立DQN模型
    json_string = open('./runs/multi_turn_qq/11.07/1478707726/model_1_pickle.json', 'r').read()
    dqn = model_from_json(json_string)
    dqn.load_weights('./runs/multi_turn_qq/11.07/1478707726/model_1_weights_sub.h5')
    
    demo_wb = Workbook()
    demo_ws = demo_wb.active
    demo_ws.title = 'Demo_Multi_Uni'

    row = 1
    demo_ws['A' + str(row)].value = 'Chat_Id'
    demo_ws['B' + str(row)].value = 'Turn_Id'
    demo_ws['C' + str(row)].value = 'Query'
    demo_ws['D' + str(row)].value = 'Answr'
    demo_ws['E' + str(row)].value = 'Quest_Mul'
    demo_ws['F' + str(row)].value = 'Qvalue'
    demo_ws['G' + str(row)].value = 'Answr_Mul'
    demo_ws['H' + str(row)].value = 'Quest_Uni'
    demo_ws['I' + str(row)].value = 'Score'
    demo_ws['J' + str(row)].value = 'Answr_Uni'

    state_num = 4
    row = 2
    chat_id = 0
    for chat in chats:
        i = 0
        chat_id += 1
        turn_id = 0
        state = deque()
        for k in range(state_num):
            state.append("")
        while i < len(chat):
            query, candi_QAs = chat[i]
            i += 1
            turn_id += 1
            answr = chat[i]
            i += 1
            mul_act, qval = choose_action_mul(state, candi_QAs.keys(), dqn)
            update_state(state, query, answr)
            uni_act, scr = choose_action_uni(candi_QAs)
            if len(uni_act) == 0:
                demo_ws['A' + str(row)].value = chat_id
                demo_ws['B' + str(row)].value = turn_id
                demo_ws['C' + str(row)].value = query
                demo_ws['D' + str(row)].value = answr
                demo_ws['E' + str(row)].value = mul_act
                demo_ws['F' + str(row)].value = qval[0][0]
                demo_ws['G' + str(row)].value = candi_QAs[mul_act][1]
                demo_ws['H' + str(row)].value = None
                demo_ws['I' + str(row)].value = None
                demo_ws['J' + str(row)].value = None
                row += 1
                continue
            for j in range(len(uni_act)):
                demo_ws['A' + str(row)].value = chat_id
                demo_ws['B' + str(row)].value = turn_id
                demo_ws['C' + str(row)].value = query
                demo_ws['D' + str(row)].value = answr
                demo_ws['E' + str(row)].value = mul_act
                demo_ws['F' + str(row)].value = qval[0][0]
                demo_ws['G' + str(row)].value = candi_QAs[mul_act][1]
                demo_ws['H' + str(row)].value = uni_act[j]
                demo_ws['I' + str(row)].value = scr[j]
                demo_ws['J' + str(row)].value = candi_QAs[uni_act[j]][1]
                row += 1
    demo_wb.save('demo_multi_uni.xlsx')

if __name__ == '__main__':
    test_num = 30
    data_file = 'multi_turn_chat_data_del.xlsx'
    sheet_name = 'multi_turn_chats'
    chats = fetch_chats(data_file, sheet_name)
    demo_multi_uni(chats[-test_num:])
