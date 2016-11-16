#！ -*- encoding:utf-8 -*-

from keras.layers import Input, Dense, Dropout, Convolution1D, merge
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from calculate_reward import cal_reward
import numpy as np
import random
import os
import time
from calculate_reward import sent_embedding

# Hyper Parameters
ini_epsilon = 0.5          # 初始的exploration概率
fnl_epsilon = 0.01         # 最后的exploration概率
cur_epsilon = ini_epsilon  # 当前运行时的exploration概率
context_dim = 4            # 上下文的句子数
sequence_len = 30          # 每个句子固定的长度
embedding_dim = 300        # 词向量维数
filter_sizes = [1, 2]
num_filters = 100
nb_epoch = 10
batch_size = 10
drop = 0.5
samples_num = 100

#######################################
# 行为选择策略,返回行为的下标
#######################################
"""
def egreedy_action_b(query, state, candi_quests):
    actions = candi_quests.keys()
    action_dim = len(actions)
    scores = [candi_quests[action][0] for action in actions]
    ind = np.array(scores) > 80
    uni_actions = np.array(actions)[ind]
    # 逐步减小exploration的概率
    global cur_epsilon
    if random.random() < cur_epsilon:
        act_num = random.randint(0, action_dim-1)
    else:
        rewards = [cal_reward(state,query, candi_quests,action) for action in actions]
        act_num = np.argmax(rewards)
    if cur_epsilon > fnl_epsilon:
        cur_epsilon -= 0.001
    return (actions[act_num], uni_actions)
"""

def egreedy_action(query, state, dqn, candi_quests):
    actions = candi_quests.keys()
    action_dim = len(actions)
    global cur_epsilon
    if random.random() < cur_epsilon:
        act_num = random.randint(0, action_dim-1)
    else:
        state_vec = np.concatenate(np.array([sent_embedding(sent) for sent in state]), axis=1)
        actions_vec = [np.array(sent_embedding(action)) for action in actions]
        Qs = [dqn.predict([state_vec, action_vec]) for action_vec in actions_vec]
        act_num = np.argmax(Qs)
    if cur_epsilon > fnl_epsilon:
        cur_epsilon -= 0.001
    return actions[act_num]

#######################################
# 更新状态
#######################################
def update_state(state, query, answer):
    state.popleft()
    state.popleft()
    state.append(query)
    state.append(answer)
    

#######################################
# 创建dqn神经网络
#######################################
def create_Q_network():
    inputs0 = Input(shape=(sequence_len*context_dim, embedding_dim))
    inputs1 = Input(shape=(sequence_len, embedding_dim))
    conv0 = Convolution1D(num_filters, filter_sizes[0], border_mode='valid',
        init='normal', activation='relu')(inputs0)
    conv0 = Convolution1D(num_filters, filter_sizes[1], border_mode='valid',
        init='normal', activation='relu')(conv0)
    conv1 = Convolution1D(num_filters, filter_sizes[0], border_mode='valid',
        init='normal', activation='relu')(inputs1)
    conv1 = Convolution1D(num_filters, filter_sizes[1], border_mode='valid',
        init='normal', activation='relu')(conv1)
    input_len0 = sum([sequence_len*context_dim - filter + 1 for filter in filter_sizes])
    forwards_vanilla0 = GRU(300, input_shape=(input_len0, embedding_dim),
        return_sequences=False, activation='relu', consume_less='gpu')(conv0)
    backwards_vanilla0 = GRU(300, input_shape=(input_len0, embedding_dim),
        return_sequences=False, go_backwards=True, activation='relu', consume_less='gpu')(conv0)
    merged_vanilla0 = merge([forwards_vanilla0, backwards_vanilla0], mode='concat', concat_axis=-1)
    input_len1 = sum([sequence_len - filter + 1 for filter in filter_sizes])
    forwards_vanilla1 = GRU(300, input_shape=(input_len1, embedding_dim),
        return_sequences=False, activation='relu', consume_less='gpu')(conv1)
    backwards_vanilla1 = GRU(300, input_shape=(input_len1, embedding_dim),
        return_sequences=False, go_backwards=True, activation='relu', consume_less='gpu')(conv1)
    merged_vanilla1 = merge([forwards_vanilla1, backwards_vanilla1], mode='concat', concat_axis=-1)
    merged_tensor = merge([merged_vanilla0, merged_vanilla1], mode='concat', concat_axis=-1)
    hidden = Dense(1024, activation='relu')(merged_tensor)
    dropout = Dropout(drop)(hidden)
    output = Dense(1, activation='relu')(dropout)
    model = Model(input=[inputs0, inputs1], output=output)
    return model


#######################################
# 训练dqn神经网络
#######################################
def train_Q_network(dqn, replay_buffer):
    target_Q = []
    state_buff = []
    action_buff = []
    for context in replay_buffer:
        # 2.2.1 依据当前状态用神经网络Q计算Q值Q_value
        state = context.popleft()
        state_vec = np.concatenate(np.array([sent_embedding(sent) for sent in state]), axis=1)
        state_buff.append(state_vec)
        action = context.popleft()
        action_vec = np.array(sent_embedding(action))
        action_buff.append(action_vec)
        # 2.2.2 用reward和更新后的状态计算目标Q值target_Q
        reward = context.popleft()
        next_state = context.popleft()
        if next_state[2] == "" and next_state[3] == "":
            tar_Q = reward
        else:
            next_state_vec = np.concatenate(np.array([sent_embedding(sent) for sent in next_state]), axis=1)
            tar_Q = reward + dqn.predict([next_state_vec, action_vec])
        target_Q.append(tar_Q)
    # target_Q = np.concatenate(np.array(target_Q), axis=0)
    action_buff = np.concatenate(np.array(action_buff), axis=0)
    target_Q = np.array(target_Q)
    y_train = target_Q[:-samples_num/10]
    y_valid = target_Q[-samples_num/10:]
    state_buff = np.concatenate(np.array(state_buff), axis=0)
    X_train = state_buff[:-samples_num/10]
    X_valid = state_buff[-samples_num/10:]
    A_train = action_buff[:-samples_num/10]
    A_valid = action_buff[-samples_num/10:]
    timestamp = str(int(time.time()))
    os.mkdir('runs/multi_turn_qq/11.07/%s' % timestamp)
    checkpoint = ModelCheckpoint(
        './runs/multi_turn_qq/11.07/%s/weights.{epoch:03d}-{val_acc:.4f}.hdf5'% timestamp,
        monitor='val_acc',verbose=1, save_best_only=True, mode='max')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    dqn.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    open('./runs/multi_turn_qq/11.07/%s/model_1_pickle.json' % timestamp, 'w').write(dqn.to_json())
    dqn.fit([X_train,A_train], y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,callbacks=[checkpoint],
        validation_data=([X_valid,A_valid], y_valid))
    dqn.save_weights('./runs/multi_turn_qq/11.07/%s/model_1_weights_sub.h5' % timestamp)


"""
if __name__ == '__main__':
    from calculate_reward import sent_embedding
    state = [u'你好啊',
             u'你好',
             u'吃饭了吗',
             u'吃过了']
    query =  u'吃了什么'
    candi_quests = {u'你吃什么了':[u'吃了米饭和鸡块', u'去吃了肯德基'],
                    u'你想吃什么':[u'吃点炒饭吧', u'还不太饿，不想吃东西'],
                    u'你喜欢吃什么':[u'水果和蔬菜'],
                    u'你吃了什么':[u'吃了碗大排面', u'吃了些酸奶和饼干', u'还没吃呢'],
                    u'你干了什么':[u'做作业了', u'看了会电视']}
    action = u'你吃什么了'
    act = egreedy_action(query, state, candi_quests)
    print act
    model = create_Q_network()
    state_vec = np.concatenate(np.array([sent_embedding(sent) for sent in state]), axis=1)
    print state_vec.shape
    print model.predict([state_vec])
"""
