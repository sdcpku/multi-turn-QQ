# coding=utf-8

import numpy as np
import codecs
import random
import itertools
from collections import Counter
from nlp_helpers import remove_stop_word
from gensim.models import Word2Vec


def load_data_and_labels_without_shuffled():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    with codecs.open('./data/train_pos.txt', 'r+', 'utf-8') as f:
        train_pos = f.readlines()
    with codecs.open('./data/dev_pos.txt', 'r+', 'utf-8') as f:
        dev_pos = f.readlines()
    with codecs.open('./data/train_neg.txt', 'r+', 'utf-8') as f:
        train_neg = f.readlines()
    with codecs.open('./data/dev_neg.txt', 'r+', 'utf-8') as f:
        dev_neg = f.readlines()

    positive_examples1 = []
    positive_examples2 = []
    negative_examples1 = []
    negative_examples2 = []

    for i in train_pos:
        item1, item2 = i.split('\t')
        item1 = remove_stop_word(item1)
        item2 = remove_stop_word(item2)
        positive_examples1.append(item1)
        positive_examples2.append(item2)

    for i in train_neg:
        item1, item2 = i.split('\t')
        item1 = remove_stop_word(item1)
        item2 = remove_stop_word(item2)
        negative_examples1.append(item1)
        negative_examples2.append(item2)

    # Split by words
    x_text_train1 = positive_examples1 + negative_examples1
    x_text_train2 = positive_examples2 + negative_examples2

    positive_dev1 = []
    positive_dev2 = []
    negative_dev1 = []
    negative_dev2 = []

    for i in dev_pos:
        item1, item2 = i.split('\t')
        item1 = remove_stop_word(item1)
        item2 = remove_stop_word(item2)
        positive_dev1.append(item1)
        positive_dev2.append(item2)

    for i in dev_neg:
        item1, item2 = i.split('\t')
        item1 = remove_stop_word(item1)
        item2 = remove_stop_word(item2)
        negative_dev1.append(item1)
        negative_dev2.append(item2)

    x_text_dev1 = positive_dev1 + negative_dev1
    x_text_dev2 = positive_dev2 + negative_dev2

    # Generate labels
    train_positive_labels = [[0, 1] for _ in train_pos]
    dev_positive_labels = [[0, 1] for _ in dev_pos]
    train_negative_labels = [[1, 0] for _ in train_neg]
    dev_negative_labels = [[1, 0] for _ in dev_neg]
    y_train = np.concatenate([train_positive_labels, train_negative_labels], 0)
    y_dev = np.concatenate([dev_positive_labels, dev_negative_labels], 0)

    return [x_text_train1, x_text_train2, x_text_dev1, x_text_dev2, y_train, y_dev]


def split_sentence(sentences):
    modified_sentences = [s.replace('\r\n', '').split(' ') for s in sentences]
    return modified_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))  # 实际没用到
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # 加入 <UNK>
    vocabulary_inv.insert(0, '</s>')
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    # !!! 一定要注意这里会影响数据的形状,要与代码内的 sequence length 保持一致 !!!
    sequence_length = 30
    # sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i][:sequence_length]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    count = 0
    seq2seq_sentences = []
    for sentence in sentences:
        seq2seq_sentence = []
        for word in sentence:
            try:
                seq2seq_sentence.append(vocabulary[word])
            except KeyError:
                seq2seq_sentence.append(vocabulary['</s>'])
                count += 1
        seq2seq_sentences.append(seq2seq_sentence)
    print count
    return np.array(seq2seq_sentences)


# TODO: Put it globally is actually dangerous, but faster
print 'Load word2vec ...'
FILE_NAME = '/Users/chenxiulong/Downloads/vectors.bin.0805.final'
model = Word2Vec.load_word2vec_format(FILE_NAME, binary=True, unicode_errors='ignore')
print 'Done'


def sentence_word2vec(sentences):
    # print 'Load word2vec ...'
    # FILE_NAME = './vectors.bin.skipgram.mergenew.2.3'
    # model = Word2Vec.load_word2vec_format(FILE_NAME, binary=True, unicode_errors='ignore')

    embedding = np.zeros((len(sentences), 30, 300))
    for i, line in enumerate(sentences):
        for j, word in enumerate(line):
            try:
                embedding[i][j] = model.syn0[model.vocab[word].index]
            except KeyError:
                embedding[i][j] = model.syn0[model.vocab['</s>'].index]
    return embedding


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    x_text_train1, x_text_train2, x_text_dev1, x_text_dev2, y_train, y_dev = load_data_and_labels_without_shuffled()

    x_text_train1 = split_sentence(x_text_train1)
    x_text_train2 = split_sentence(x_text_train2)
    x_text_dev1 = split_sentence(x_text_dev1)
    x_text_dev2 = split_sentence(x_text_dev2)

    x_text_train1 = pad_sentences(x_text_train1)
    x_text_train2 = pad_sentences(x_text_train2)
    x_text_dev1 = pad_sentences(x_text_dev1)
    x_text_dev2 = pad_sentences(x_text_dev2)

    # sentences = x_text_train1 + x_text_train2 + x_text_dev1 + x_text_dev2
    # vocabulary, vocabulary_inv = build_vocab(sentences)
    # x_text_train1 = build_input_data(x_text_train1, vocabulary)
    # x_text_train2 = build_input_data(x_text_train2, vocabulary)
    # x_text_dev1 = build_input_data(x_text_dev1, vocabulary)
    # x_text_dev2 = build_input_data(x_text_dev2, vocabulary)

    x_train1 = sentence_word2vec(x_text_train1)
    x_train2 = sentence_word2vec(x_text_train2)
    x_dev1 = sentence_word2vec(x_text_dev1)
    x_dev2 = sentence_word2vec(x_text_dev2)

    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    # return [x_text_train1, x_text_train2, x_text_dev1, x_text_dev2, y_train, y_dev, vocabulary, vocabulary_inv]

    return [x_train1, x_train2, x_dev1, x_dev2, y_train, y_dev]
