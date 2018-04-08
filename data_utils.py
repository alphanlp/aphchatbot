# -*- coding: utf-8 -*-

import re


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



start_token = 1
end_token = 1000


def prepare_train_batch(seqs_x, seqs_y, maxlen=None):
    # seqs_x, seqs_y: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x <= maxlen and l_y <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

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



if __name__ == '__main__':
    conv_list = load_xhj()
    print(conv_list)
