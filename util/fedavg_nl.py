# python version 3.7.1
# -*- coding: utf-8 -*-

import copy
import torch


def FedAvgNL(w, dict_len,noisy_level):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():        
        w_avg[k] = w_avg[k] * dict_len[0]*(1-noisy_level[0])
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]*(1-noisy_level[i])
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg
