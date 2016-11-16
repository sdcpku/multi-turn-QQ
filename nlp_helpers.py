# coding=utf-8

import codecs

STOP_WORD_PATH = './stopwords.txt'


def remove_stop_word(sentence):
    with codecs.open(STOP_WORD_PATH, 'r', 'utf-8') as f:
        stopword_set = set(ch.strip() for ch in f.readlines())
    sentence = ' '.join(word for word in sentence.split(' ') if word not in stopword_set)
    return sentence


if __name__ == '__main__':
    print remove_stop_word(u'我 好 饿 啊 ， 你 呢 ？')

