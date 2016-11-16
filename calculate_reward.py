#! -*- encoding:utf-8 -*-

import tensorflow as tf
tf.python.control_flow_ops = tf

from nlp_helpers import remove_stop_word
from data_helpers import pad_sentences
from data_helpers import sentence_word2vec
from data_helpers import split_sentence
import numpy as np
import numpy.linalg as ln
from math import fabs
import codecs
from keras.models import model_from_json


# 由网络和权值得到模型
json_string = open('./runs/cnn-gru/model_1_pickle.json', 'r').read()
model = model_from_json(json_string)
model.load_weights('./runs/cnn-gru/model_1_weights_sub.h5')


# 将句子规整处理好并向量化
def sent_embedding(sent):
    sent = remove_stop_word(sent)
    sent = split_sentence([sent])
    sent = pad_sentences(sent)
    sent_vec = sentence_word2vec(sent)
    return sent_vec


def cal_reward(state, query, candi_quests, action):
    qq_similar_coeff = 0.5
    easy_answer_coeff = 0.2
    inform_flow_coeff = 0.3
    r1 = qq_similar(candi_quests, action)
    r2 = easy_answer(candi_quests, action)
    r3 = inform_flow(candi_quests, state, action)
    r = r1*qq_similar_coeff + r2*easy_answer_coeff + r3*inform_flow_coeff
    return r

def qq_similar(candi_quests, action):
    r1 = candi_quests[action][0] / 100.0
    return r1


#def qq_similar(query, action):
#    vec_query = sent_embedding(query)
#    if type(action) != 'unicode':
#        return 0
#    vec_action = sent_embedding(action)
#    r1 = model.predict([vec_query,vec_action], batch_size=1)[0][0]
#    return r1


def easy_answer(candi_quest, action):
    from data_auxiliary import dull_answers
    candi_answers = candi_quest[action][1:]     # 候选问题的答案列表
    # 判断候选答案是否为空
    if candi_answers == [None]:
        return 0
    candi_ans_num = len(candi_answers)      # 候选问题的答案数
    dull_ans_num = len(dull_answers)
    r2 = 0.0
    for candi_ans in candi_answers:
        if candi_ans == None:
            candi_ans_num -= 1
            continue
        r2_sub = 0.0
        if type(candi_ans) != 'unicode':
            candi_ans_num -= 1
            if candi_ans_num == 0:
                return 0
        else:
            vec_candi_ans = sent_embedding(candi_ans)
            for dull_ans in dull_answers:
                vec_dull_ans = sent_embedding(dull_ans)
                r2_sub += model.predict([vec_candi_ans, vec_dull_ans], batch_size=1)[0][0]
            r2_sub /= dull_ans_num
        r2 += r2_sub
    r2 /= candi_ans_num
    return r2


def inform_flow(candi_quests, state, action):
    # 得到上下文信息
    p1 = state[0]
    q1 = state[1]
    p2 = state[2]
    q2 = state[3]
    candi_answers = candi_quests[action][1:]
    # 判断候选答案是否为空
    if candi_answers == [None]:
        return 0
    candi_ans_num = len(candi_answers)
    q1_coeff = 0.4
    q2_coeff = 0.6
    r3 = 0
    for candi_ans in candi_answers:
        if candi_ans == None:
            candi_ans_num -= 1
            continue
        if type(candi_ans) != 'unicode':
            candi_ans_num -= 1
            if candi_ans_num == 0:
                return 0
            continue
        vec_q1 = sent_embedding(q1)[0]
        vec_q2 = sent_embedding(q2)[0]
        vec_candi_ans = sent_embedding(candi_ans)
        print "Temp r3:",
        print np.dot(np.ndarray.flatten(vec_q1),np.ndarray.flatten(vec_candi_ans)) / (ln.norm(vec_q1)*ln.norm(vec_candi_ans)),
        print np.dot(np.ndarray.flatten(vec_q2),np.ndarray.flatten(vec_candi_ans)) / (ln.norm(vec_q2)*ln.norm(vec_candi_ans))
        r3 += -np.log(fabs(np.dot(np.ndarray.flatten(vec_q1),np.ndarray.flatten(vec_candi_ans))) / (ln.norm(vec_q1)*ln.norm(vec_candi_ans))) * q1_coeff
        r3 += -np.log(fabs(np.dot(np.ndarray.flatten(vec_q2),np.ndarray.flatten(vec_candi_ans))) / (ln.norm(vec_q2)*ln.norm(vec_candi_ans))) * q2_coeff
    r3 /= candi_ans_num
    return r3
