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


if __name__ == '__main__':
    conv_list = load_xhj()
    print(conv_list)
