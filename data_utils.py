# -*- coding: utf-8 -*-

import re
import numpy as np


def load_xhj():
    '''
    返回一个二维嵌套list
    :return:
    '''
    data_list = []
    path = './resources/xiaohuangji50w_nofenci.conv'
    with open(path, encoding='utf-8') as xhj:
        line = xhj.readline()
        count = 0
        while line:
            if line.startswith("E"):
                # count += 1
                # if count > 10:
                #     break
                a_list = []
                line = xhj.readline()
                while line and not line.startswith("E"):
                    line = re.sub(re.compile('^M\s+'), '', line)
                    a_list.append(line.strip())
                    line = xhj.readline()
                if a_list:
                    data_list.append(a_list)

    return data_list


def load_tolist(path):
    data_list = []
    with open(path, encoding='utf-8') as f:
        data_list = f.readlines()
    return data_list


extra_tokens = ['_GO', 'EOS']

start_token = extra_tokens.index('_GO')
end_token = extra_tokens.index('EOS')

input_sentence = [
    "天王盖地虎",
    "你谈过恋爱么",
    "在干嘛"
]

output_sentence = [
    "宝塔镇妖河。",
    "谈过，哎，别提了，伤心",
    "在想你啊"
]

char_to_index = {}
index_to_char = {}
vocab_size = 0

def prepare():
    global char_to_index
    global index_to_char
    global vocab_size
    text = "".join(input_sentence + output_sentence)
    char_list = list(set(text))
    vocab_size = len(char_list) + len(extra_tokens)
    char_to_index = {ch: i + len(extra_tokens) for i, ch in enumerate(char_list)}
    index_to_char = {idx: ch for idx, ch in char_to_index.items()}
    return char_to_index, index_to_char, vocab_size


def train_set():
    for input, output in zip(input_sentence, output_sentence):
        form_input = []
        form_output = []
        for ch in input:
            form_input.append(char_to_index[ch])
        for ch in output:
            form_output.append(char_to_index[ch])
        form_output.append(end_token)
        yield [form_input], [form_output]


def prepare_train_batch(seqs_x, seqs_y, maxlen=None):
    # seqs_x, seqs_y: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    batch_size = len(seqs_x)

    x_lengths = np.array(lengths_x)
    y_lengths = np.array(lengths_y)

    maxlen_x = np.max(x_lengths)
    maxlen_y = np.max(y_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * end_token
    y = np.ones((batch_size, maxlen_y)).astype('int32') * end_token

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[idx, :lengths_x[idx]] = s_x
        y[idx, :lengths_y[idx]] = s_y
    return x, x_lengths, y, y_lengths


def prepare_predict_batch(seqs_x, maxlen=None):
    lengths_x = [len(s) for s in seqs_x]

    batch_size = len(seqs_x)
    x_lengths = np.array(lengths_x)
    maxlen_x = np.max(x_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * end_token
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, x_lengths

if __name__ == '__main__':
    conv_list = load_xhj()
    print(conv_list)
