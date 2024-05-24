# python version 3.7.1
# -*- coding: utf-8 -*-

import copy
import torch


def PerFedAvg(w,w_last, dict_len):
    w_avg1 = copy.deepcopy(w[0])
    w_avg2 = copy.deepcopy(w_last[0])
    for k in w_avg1.keys():
        w_avg1[k] = w_avg1[k] * dict_len[0]*0.7 + w_avg2[k] * dict_len[0]*0.3
        for i in range(1, len(w)):
            w_avg1[k] += w[i][k] * dict_len[i] * 0.7 + w_last[i][k] * dict_len[i]*0.3
        w_avg1[k] = w_avg1[k] / sum(dict_len)
    return w_avg1
