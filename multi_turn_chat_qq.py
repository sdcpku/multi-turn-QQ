#! -*- coding:utf-8 -*-

# 使用tensorflow rc0.11时需加上下面两行代码
import tensorflow as tf
tf.python.control_flow_ops = tf

from collections import deque
import numpy as np
from dqn_cnn_rnn import create_Q_network
from dqn_cnn_rnn import train_Q_network
from dqn_cnn_rnn import egreedy_action      # 依据epsilon-greedy策略选择行为的函数
from dqn_cnn_rnn import update_state
from calculate_reward import cal_reward     # 由行为action计算奖励reward
from calculate_reward import sent_embedding
from data_auxiliary import fetch_chats
from data_auxiliary import display_context
# from demo_multi_uni import demo_multi_uni

# 获取数据
data_file = 'multi_turn_chat_data_del.xlsx'
sheet_name = 'multi_turn_chats'
chats = fetch_chats(data_file, sheet_name)

# 1. 初始化
replay_buffer = []              # 经验池，存储每次优化时的单元
dqn = create_Q_network()        # 初始化神经网络

cnt = 0
# 2 训练每一个chat
for chat in chats:
    # 2.1 初始化状态
    state_num = 4
    state = deque(maxlen=state_num)
    for i in range(state_num):
        state.append("")
    action = ""
    reward = 0
    # 2.2 计算每个episode的对话轮turn
    i = 0
    while i < len(chat):
        context = deque()
        context.append(state)
        context.append(action)
        context.append(reward)
        query, quest_answrs = chat[i]
        i += 1
        answer = chat[i]
        # 2.2.1 依据策略选择一个行为action
        action = egreedy_action(query, state, dqn, quest_answrs)
        # 2.2.2 执行action，获得reward，更新状态state
        reward = cal_reward(state, query, quest_answrs, action)
        update_state(state, query, answer)
        context.append(state)
        # 2.2.3 将相关信息存储进经验池
        # display_context(context)
        replay_buffer.append(context)
        i += 1
        cnt += 1
    else:
        context = deque()
        context.append(state)
        context.append(action)
        context.append(reward)
        update_state(state, "", "")
        context.append(state)
        # display_context(context)
        replay_buffer.append(context)
        cnt += 1
    # 2.3 对Q_value和target_Q之间的MSE用SGD优化
    if cnt % 50 == 0:
        train_Q_network(dqn, replay_buffer)
        replay_buffer = []

# demo_multi_uni(chats)
