# python version 3.7.1
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
import torch.nn as nn

from util.options import args_parser
from util.local_training import LocalUpdate, globaltest
from util.fedavg import FedAvg
from util.util import add_noise, lid_term, get_output
from util.dataset import get_dataset
from model.build_model import build_model

np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""

if __name__ == '__main__':
    # parse args
    args = args_parser()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    rootpath = "./record/"

    dataset_train, dataset_test, dict_users = get_dataset(args)

    # ---------------------Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    if not os.path.exists(rootpath + 'txtsave/'):
        os.makedirs(rootpath + 'txtsave/')
    txtpath = rootpath + 'txtsave/%s_%s_NL_%.1f_LB_%.1f_Iter_%d_Rnd_%d_%d_ep_%d_Frac_%.3f_%.2f_LR_%.3f_ReR_%.1f_ConT_%.1f_ClT_%.1f_Beta_%.1f_Seed_%d' % (
        args.dataset, args.model, args.level_n_system, args.level_n_lowerb, args.iteration1, args.rounds1,
        args.rounds2, args.local_ep, args.frac1, args.frac2, args.lr, args.relabel_ratio,
        args.confidence_thres, args.clean_set_thres, args.beta, args.seed)

    if args.iid:
        txtpath += "_IID"
    else:
        txtpath += "_nonIID_p_%.1f_dirich_%.1f" % (args.non_iid_prob_class, args.alpha_dirichlet)
    if args.fine_tuning:
        txtpath += "_FT"
    if args.correction:
        txtpath += "_CORR"
    if args.mixup:
        txtpath += "_Mix_%.1f" % (args.alpha)

    f_acc = open(txtpath + '_acc.txt', 'a')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # build model
    netglob = build_model(args)
    net_local = build_model(args)

    client_p_index = np.where(gamma_s == 0)[0]  # 获取无噪声的客户端
    client_n_index = np.where(gamma_s > 0)[0]  # 获取含噪声的客户端
    criterion = nn.CrossEntropyLoss(reduction='none')  # 定义损失函数
    LID_accumulative_client = np.zeros(args.num_users)  # 初始化累积LID分数
    # ------------------------Preprocess Stage--------------------------
    for iteration in range(args.iteration1):  # 预处理阶段迭代
        LID_whole = np.zeros(len(y_train))  # 初始化整个训练集的LID分数
        loss_whole = np.zeros(len(y_train))  # 初始化整个训练集的损失值
        LID_client = np.zeros(args.num_users)  # 初始化各个客户端的LID分数
        loss_accumulative_whole = np.zeros(len(y_train))  # 初始化各个客户端的累积损失值
        # 上述四个玩意都在累加 TODO 确认
        # ---------Broadcast global model----------------------
        if iteration == 0:
            mu_list = np.zeros(args.num_users)
        else:
            mu_list = estimated_noisy_level  # 噪声检测水平

        prob = [1 / args.num_users] * args.num_users  # 各个用户概率 TODO 意义

        for i in range(int(1 / args.frac1)):  # 每个用户都能被选中一次 TODO 一层循环即可
            idxs_users = np.random.choice(range(args.num_users), int(args.num_users * args.frac1), p=prob)  # 随机选择一部分用户
            w_locals = []  # 本地模型权重列表
            for idx in idxs_users:  # 对于选中的用户进行训练
                prob[idx] = 0  # 在本次迭代中概率设为0，不会重复选择同一用户
                if sum(prob) > 0:  # 如果还有剩余的用户
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]  # 更新概率(也是为了保证被选到)

                net_local.load_state_dict(netglob.state_dict())  # 将本地模型加载为全局模型
                sample_idx = np.array(list(dict_users[idx]))  # 获取用户对应的样本索引
                dataset_client = Subset(dataset_train, sample_idx)  # 获取用户对应的子数据集
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)

                # proximal term operation
                mu_i = mu_list[idx]  # 获取当前用户的噪声检测水平
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)  # 创建本地更新对象
                w, loss = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=args.seed,
                                               w_g=netglob.to(args.device), epoch=args.local_ep,
                                               mu=mu_i)  # 获取更新后的模型权重和损失

                net_local.load_state_dict(copy.deepcopy(w))  # 更新本地模型
                w_locals.append(copy.deepcopy(w))  # 将更新后的模型权重添加到列表中
                acc_t = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)  # 测试本地模型在测试集上的准确率
                f_acc.write("iteration %d, client %d, acc: %.4f \n" % (iteration, idx, acc_t))  # 写入准确率
                print("iteration {}, client {}, acc: {:.4f}".format(iteration, idx, acc_t))  # 打印准确率
                f_acc.flush()

                local_output, loss = get_output(loader, net_local.to(args.device), args, False,
                                                criterion)  # 获取本地模型在训练集上的输出和损失值（交叉熵）
                LID_local = list(lid_term(local_output, local_output))  # 计算本地模型的LID分数
                LID_whole[sample_idx] = LID_local  # 更新样本的LID分数
                loss_whole[sample_idx] = loss  # 更新样本的损失值
                LID_client[idx] = np.mean(LID_local)  # 计算客户端的平均LID分数
            # 客户端全部遍历一遍后
            dict_len = [len(dict_users[idx]) for idx in idxs_users]  # 计算每个客户端的样本数 TODO 计算一次即可
            w_glob = FedAvg(w_locals, dict_len)  # 全局权重更新

            netglob.load_state_dict(copy.deepcopy(w_glob))  # 全局模型更新

        LID_accumulative_client = LID_accumulative_client + np.array(LID_client)  # 更新各个客户端的累积LID分数
        loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)  # 更新各个客户端的累积损失值

        # Apply Gaussian Mixture Model to LID
        gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=args.seed).fit(
            np.array(LID_accumulative_client).reshape(-1, 1))  # 使用GMM聚类累积客户端的LID值
        labels_LID_accumulative = gmm_LID_accumulative.predict(
            np.array(LID_accumulative_client).reshape(-1, 1))  # 使用GMM对客户端进行标记
        clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]  # 获取干净客户端的索引

        noisy_set = np.where(labels_LID_accumulative != clean_label)[0]  # 获取噪声客户端索引
        clean_set = np.where(labels_LID_accumulative == clean_label)[0]  # 获取干净数据集的索引

        estimated_noisy_level = np.zeros(args.num_users)

        for client_id in noisy_set:  # 迭代噪声客户端
            sample_idx = np.array(list(dict_users[client_id]))  # 获取噪声客户端中的样本索引

            loss = np.array(loss_accumulative_whole[sample_idx])  # 获取各个样本的累积损失值
            gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(
                np.array(loss).reshape(-1, 1))  # 用GMM对各个样本的累积损失值进行聚类
            labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))  # 对各个样本打上标签
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]  # 排序样本，损失值更小的打上干净标签

            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]  # 得到各个客户端中的噪声样本索引
            estimated_noisy_level[client_id] = len(pred_n) / len(sample_idx)  # 估计各客户端的噪声水平
            y_train_noisy_new = np.array(dataset_train.targets)  # TODO 冗余

        if args.correction:  # 如果进行标签修正
            for idx in noisy_set:  # 对噪声客户端进行迭代
                sample_idx = np.array(list(dict_users[idx]))  # 获取该客户端的样本标签
                dataset_client = Subset(dataset_train, sample_idx)  # 获取该客户端的子数据集
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                loss = np.array(loss_accumulative_whole[sample_idx])  # 获取该客户端训练样本的损失累积值
                local_output, _ = get_output(loader, netglob.to(args.device), args, False,
                                             criterion)  # 在全局模型上获取客户端的预测输出结果
                relabel_idx = (-loss).argsort()[:int(
                    len(sample_idx) * estimated_noisy_level[
                        idx] * args.relabel_ratio)]  # 对累积损失越大的进行排序，选出一定数量的标签待修改样本，数量由估计噪声水平以及重标记比例决定
                relabel_idx = list(
                    set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(
                        relabel_idx))  # 如果输出置信度也大于阈值则为待修改样本

                y_train_noisy_new = np.array(dataset_train.targets)  # 获取新的训练集标签
                y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[
                    relabel_idx]  # 用全局模型输出结果更新标签
                dataset_train.targets = y_train_noisy_new  # 更新数据集标签

    # reset the beta,
    args.beta = 0
    print('pre finished')
    # ---------------------------- second stage training -------------------------------
    if args.fine_tuning:  # 如果进行微调
        selected_clean_idx = np.where(estimated_noisy_level <= args.clean_set_thres)[0]  # 获取干净数据集的索引

        prob = np.zeros(args.num_users)  # 初始化概率为0
        prob[selected_clean_idx] = 1 / len(selected_clean_idx)  # 设置干净数据集的概率
        m = max(int(args.frac2 * args.num_users), 1)  # 计算选择的客户端数量
        m = min(m, len(selected_clean_idx))  # 选择最小值
        netglob = copy.deepcopy(netglob)  # 拷贝全局模型
        # add fl training
        for rnd in range(args.rounds1):  # 迭代微调的轮数
            w_locals, loss_locals = [], []  # 初始化本地模型权重列表和本地模型损失列表
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)  # 随机选择一部分客户端
            for idx in idxs_users:  # 对于每个客户端
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # 创建本地更新对象
                w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                           w_g=netglob.to(args.device), epoch=args.local_ep,
                                                           mu=0)  # 更新本地模型权重
                w_locals.append(copy.deepcopy(w_local))  # 将更新后的模型权重添加到列表中
                loss_locals.append(copy.deepcopy(loss_local))  # 将损失添加到列表中

            dict_len = [len(dict_users[idx]) for idx in idxs_users]  # 计算每个客户端的样本数
            w_glob_fl = FedAvg(w_locals, dict_len)  # 使用FedAvg算法进行全局模型更新
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))  # 更新全局模型权重

            acc_s2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)  # 测试全局模型在测试集上的准确率
            f_acc.write("fine tuning stage round %d, test acc  %.4f \n" % (rnd, acc_s2))  # 将准确率写入文件
            print("fine tuning stage round {}, test acc {:.4f}".format(rnd, acc_s2))  # 打印准确率
            f_acc.flush()  # 刷新文件

        if args.correction:  # 如果进行纠正
            relabel_idx_whole = []  # 初始化纠正的索引列表
            for idx in noisy_set:  # 对于噪声数据集中的每个客户端
                sample_idx = np.array(list(dict_users[idx]))  # 获取对应的样本索引
                dataset_client = Subset(dataset_train, sample_idx)  # 获取用户对应的子数据集
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)  # 创建数据加载器
                glob_output, _ = get_output(loader, netglob.to(args.device), args, False,
                                            criterion)  # 获取全局模型在训练集上的输出和损失
                y_predicted = np.argmax(glob_output, axis=1)  # 获取预测结果
                relabel_idx = np.where(np.max(glob_output, axis=1) > args.confidence_thres)[0]  # 获取需要纠正的索引
                y_train_noisy_new = np.array(dataset_train.targets)  # 获取新的训练集标签
                y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]  # 更新标签
                dataset_train.targets = y_train_noisy_new  # 更新数据集标签

    # ---------------------------- third stage training -------------------------------
    # third stage hyper-parameter initialization
    m = max(int(args.frac2 * args.num_users), 1)  # 计算选择的客户端数量
    prob = [1 / args.num_users for i in range(args.num_users)]  # 初始化每个用户的概率相等

    for rnd in range(args.rounds2):  # 迭代第三阶段的轮数
        w_locals, loss_locals = [], []  # 初始化本地模型权重列表和本地模型损失列表
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)  # 随机选择一部分客户端
        for idx in idxs_users:  # 对于每个客户端
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # 创建本地更新对象
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                       w_g=netglob.to(args.device), epoch=args.local_ep,
                                                       mu=0)  # 更新本地模型权重
            w_locals.append(copy.deepcopy(w_local))  # 将更新后的模型权重添加到列表中
            loss_locals.append(copy.deepcopy(loss_local))  # 将损失添加到列表中

        dict_len = [len(dict_users[idx]) for idx in idxs_users]  # 计算每个客户端的样本数
        w_glob_fl = FedAvg(w_locals, dict_len)  # 使用FedAvg算法进行全局模型更新
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))  # 更新全局模型权重

        acc_s3 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)  # 测试全局模型在测试集上的准确率
        f_acc.write("round %d, test acc  %.4f \n" % (rnd, acc_s3))  # 将准确率写入文件
        print("round {}, test acc {:.4f}".format(rnd, acc_s3))  # 打印准确率
        f_acc.flush()  # 刷新文件

    torch.cuda.empty_cache()
