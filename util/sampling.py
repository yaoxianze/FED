# python version 3.7.1
# -*- coding: utf-8 -*-
import numpy as np


def iid_sampling(n_train, num_users, seed):  # 生成IID数据样本
    np.random.seed(seed)  # 使用参数中固定得随机数生成器的种子，确保每次运行代码时得到的随机结果是一致的
    num_items = int(n_train / num_users)  # 计算每个用户获得的样本个数
    dict_users, all_idxs = {}, [i for i in range(n_train)]  # dict_users用于存储每个用户抽取的样本索引集合，all_idxs用于储存一个包含整个数据集索引的列表
    for i in range(num_users):
        dict_users[i] = set(  # 从all_idxs中随机选择num_items个样本索引，并将其存储为集合形式
            np.random.choice(all_idxs, num_items, replace=False))  # 'replace=False' make sure that there is no repeat
        all_idxs = list(set(all_idxs) - dict_users[i])  # 每个用户选择后在整个数据集索引中去除掉
    return dict_users


def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    np.random.seed(seed)
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # 建造一个i行j列的01矩阵，其中的项表示第i个用户是否选择类别j
    n_classes_per_client = np.sum(Phi, axis=1)  # 统计每个用户选择的类别数
    while np.min(n_classes_per_client) == 0:  # 如果有任何一个用户没有划分到任何类别，则重新随机选择类别，直到每个客户端至少选择一个类别。
        invalid_idx = np.where(n_classes_per_client == 0)[0]  # 没有划分到类别的用户
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))  # 没有划分到类别的用户再次进行划分
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j] == 1)[0]) for j in range(num_classes)]  # 一个列表，表中第j个元素表示分配到第j个类别的用户索引
    num_clients_per_class = np.array([len(x) for x in Psi])  # 统计各个类别的用户数
    dict_users = {}
    for class_i in range(num_classes):  # 对各个样本类别进行分配
        all_idxs = np.where(y_train == class_i)[0]  # 某种类别的所有训练样本的行索引
        p_dirichlet = np.random.dirichlet(
            [alpha_dirichlet] * num_clients_per_class[class_i])  # 使用狄利克雷分布随机生成一个概率向量，表示每个客户端在该类别上的权重
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())  #

        for client_k in Psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])
    return dict_users
