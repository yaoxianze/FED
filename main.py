# python version 3.7.1
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from sklearn.metrics import confusion_matrix
import os
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from util.util import calculate_metrics
from torch.utils.data import Subset
from sklearn.mixture import GaussianMixture
import torch.nn as nn
import scipy.stats as stats
from scipy.stats import gaussian_kde
from util.options import args_parser
from util.local_training import LocalUpdate, globaltest
from util.fedavg import FedAvg
from util.fedavg_nl import FedAvgNL
from util.per_fed import PerFedAvg
from util.util import add_noise, lid_term, get_output
from util.dataset import get_dataset
from util.util import SCELoss
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

    sample_client = []
    for client_idx, sample_idxs in dict_users.items():
        sample_count = len(sample_idxs)
        sample_client.append(sample_count)
    print("各个客户端的样本个数：", sample_client)

    # ---------------------Add Noise ---------------------------
    y_train = np.array(dataset_train.targets)
    y_train_noisy, gamma_s, real_noise_level = add_noise(args, y_train, dict_users)
    dataset_train.targets = y_train_noisy

    whether_noisy = [1 if x == y else 0 for x, y in zip(y_train, y_train_noisy)]  # 样本是否为噪声样本,干净为1.噪声为0
    whether_noisy = np.array(whether_noisy)
    # print(whether_noisy.shape)
    # plot confusion matrix (take the first 5 clients for example)
    fig, axes = plt.subplots(1, 5, sharex=False, sharey=True, figsize=(18, 3), dpi=600)
    for i, ax in enumerate(axes):
        idx = list(dict_users[i])
        y_true = y_train[idx]
        y_noisy = y_train_noisy[idx]
        conf_matrix = confusion_matrix(y_true, y_noisy)
        im = ax.imshow(conf_matrix, cmap=plt.cm.hot_r)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title("Client {}".format(i + 1))  # 设置每个子图的标题为“客户端 i”
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)
        plt.colorbar(im, cax=cax)
    # 设置整个图的标签
    fig.text(0.5, 0.01, 'Corret Label', ha='center', fontsize=14)
    fig.text(0.1, 0.5, 'Noisy Label', va='center', rotation='vertical', fontsize=14)
    plt.savefig("img/noise_stat1.jpg", bbox_inches='tight')

    if not os.path.exists(rootpath + 'txtsave/'):
        os.makedirs(rootpath + 'txtsave/')
    txtpath = rootpath + 'txtsave/finaltrial%s_%s_NL_%.1f_LB_%.1f_Iter_%d_Rnd_%d_%d_ep_%d_Frac_%.3f_%.2f_LR_%.3f_ReR_%.1f_ConT_%.1f_ClT_%.1f_Beta_%.1f_Seed_%d' % (
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
    bestnetglob = build_model(args)
    netglob = build_model(args)
    net_local = build_model(args)

    client_p_index = np.where(gamma_s == 0)[0]
    client_n_index = np.where(gamma_s > 0)[0]
    criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=args.num_classes)
    LID_accumulative_client = np.zeros(args.num_users)
    accuracy = []
    accuracy3 = []

    for iteration in range(args.iteration1):
        LID_whole = np.zeros(len(y_train))
        loss_whole = np.zeros(len(y_train))
        LID_client = np.zeros(args.num_users)
        loss_accumulative_whole = np.zeros(len(y_train))

        # ---------Broadcast global model----------------------
        if iteration == 0:
            mu_list = np.zeros(args.num_users)
        else:
            mu_list = estimated_noisy_level

        prob = [1 / args.num_users] * args.num_users

        for _ in range(int(1 / args.frac1)):
            idxs_users = np.random.choice(range(args.num_users), int(args.num_users * args.frac1), p=prob)
            w_locals = []
            for idx in idxs_users:
                prob[idx] = 0
                if sum(prob) > 0:
                    prob = [prob[i] / sum(prob) for i in range(len(prob))]

                net_local.load_state_dict(netglob.state_dict())
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)

                # proximal term operation
                mu_i = mu_list[idx]
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=sample_idx)
                w, loss = local.update_weights(net=copy.deepcopy(net_local).to(args.device), seed=args.seed,
                                               w_g=netglob.to(args.device), epoch=args.local_ep, mu=mu_i)

                net_local.load_state_dict(copy.deepcopy(w))
                w_locals.append(copy.deepcopy(w))
                acc_t = globaltest(copy.deepcopy(net_local).to(args.device), dataset_test, args)
                f_acc.write("iteration %d, client %d, acc: %.4f \n" % (iteration, idx, acc_t))
                print("iteration {}, client {}, acc: {:.4f}".format(iteration, idx, acc_t))  # 打印准确率
                f_acc.flush()
                # accuracy.append(acc_t)

                local_output, loss = get_output(loader, net_local.to(args.device), args, False, criterion)
                LID_local = list(lid_term(local_output, local_output)) # local_output
                LID_whole[sample_idx] = LID_local
                loss_whole[sample_idx] = loss
                LID_client[idx] = np.mean(LID_local)
                # print(iteration,"客户端编号",idx,"Lid值",LID_client[idx],"噪声水平",real_noise_level[idx])

            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            w_glob = FedAvg(w_locals, dict_len)
            netglob.load_state_dict(copy.deepcopy(w_glob))


        # print(LID_client)
        # print("累积",LID_accumulative_client)
        LID_accumulative_client = LID_accumulative_client + np.array(LID_client)
        loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)

        fig_path = 'img/relation_acc_' + str(iteration) + '.jpg'
        g = sns.jointplot(x=LID_accumulative_client, y=real_noise_level, kind='reg')
        r, p = stats.pearsonr(LID_accumulative_client, real_noise_level)
        phantom, = g.ax_joint.plot([], [], linestyle="", alpha=0)
        # here graph is not a ax but a joint grid, so we access the axis through ax_joint method
        g.ax_joint.legend([phantom], ['pearsonr={:f}, p-value={:f}'.format(r, p)])
        g.set_axis_labels("LID", "noise level")
        g.savefig(fig_path, dpi=600)

        # Apply Gaussian Mixture Model to LID
        gmm_LID_accumulative = GaussianMixture(n_components=2, random_state=args.seed).fit(
            np.array(LID_accumulative_client).reshape(-1, 1))
        labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_accumulative_client).reshape(-1, 1))
        clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

        noisy_set = np.where(labels_LID_accumulative != clean_label)[0]
        clean_set = np.where(labels_LID_accumulative == clean_label)[0]
        print(noisy_set)
        print("上噪声，下干净")
        print(clean_set)

        # print(LID_accumulative_client)

        # GMM聚类后两类分开情况
        fig = plt.figure()
        Lid_noisy_clients = [LID_accumulative_client[i] for i in noisy_set]  # 提取噪声客户端的LID
        Lid_clean_clients = [LID_accumulative_client[i] for i in clean_set]  # 提取干净客户端的LID
        # print(Lid_noisy_clients)
        # print("上噪声，下干净")
        # print(Lid_clean_clients)
        sns.histplot(Lid_noisy_clients, alpha=0.5, label='noisy', color="red", stat="density",kde=True)
        sns.histplot(Lid_clean_clients, alpha=0.5, label='clean', color="blue", stat="density",kde=True)
        # sns.displot(Lid_noisy_clients,alpha=0.5, label='noisy', color="red", kind='kde')
        # sns.displot(Lid_clean_clients, alpha=0.5, label='clean', color="blue", kind='kde')
        sns.kdeplot(Lid_noisy_clients, color="red", label='noisy')
        sns.kdeplot(Lid_clean_clients, color="blue", label='clean')
        # sns.kdeplot(LID_accumulative_client,color="black", label='GMM')
        # plt.hist(LID_accumulative_client, bins=100,color="green")
        # sns.histplot(LID_accumulative_client, kde=True , color="green", stat= "probability")
        plt.ylabel("density")
        plt.xlabel("LID")
        plt.legend(loc='upper right')
        plt.savefig("img/GMM_" + str(iteration) + ".jpg", bbox_inches='tight')

        estimated_noisy_level = np.zeros(args.num_users)

        TP_num = np.zeros(len(noisy_set)+len(clean_set))
        FP_num = np.zeros(len(noisy_set)+len(clean_set))
        TN_num = np.zeros(len(noisy_set)+len(clean_set))
        FN_num = np.zeros(len(noisy_set)+len(clean_set))
        for client_id in noisy_set:
            sample_idx = np.array(list(dict_users[client_id]))
            loss = np.array(loss_accumulative_whole[sample_idx])
            gmm_loss = GaussianMixture(n_components=2, random_state=args.seed).fit(np.array(loss).reshape(-1, 1))
            labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
            gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]
            # print("噪声样本标签",gmm_clean_label_loss)
            pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0] #噪声样本索引
            pred_c = np.where(labels_loss.flatten() == gmm_clean_label_loss)[0] #干净样本索引
            estimated_noisy_level[client_id] = len(pred_n) / len(sample_idx) # 估计客户端噪声水平
            whether_values = [whether_noisy[idx] for idx in sample_idx]
            pred_values = np.zeros(len(pred_n)+len(pred_c))
            pred_values[pred_c] = 1 # 干净为1，噪声为0
            TP, FP, TN, FN = calculate_metrics(whether_values, pred_values)
            TP_num[client_id] = TP
            FP_num[client_id] = FP
            TN_num[client_id] = TN
            FN_num[client_id] = FN
            y_train_noisy_new = np.array(dataset_train.targets)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#8ECFC9','#82B0D2','#FFBE7A', '#FA7F6F']
        x = range(len(TP_num))
        ax.bar(x, TP_num, label='TP',color=colors[0])
        ax.bar(x, FP_num, bottom=TP_num, label='FP',color = colors[1])
        ax.bar(x, TN_num, bottom=TP_num + FP_num, label='TN',color=colors[2])
        ax.bar(x, FN_num, bottom=TP_num + FP_num + TN_num, label='FN',color=colors[3])
        ax.legend()  # 添加图例
        plt.ylabel("num")
        plt.xlabel("Client index")
        plt.savefig("img/sample_test_bar_"+str(iteration)+".jpg", bbox_inches='tight')
        # print("噪声标签检测准确率",TP_num/(TP_num+FP_num))
        # print("噪声标签检测查全率", TP_num/(TP_num+FN_num))
        # 预处理阶段的标签修改效果如何
        noise_level_before = np.zeros(len(real_noise_level))  # 存储每个客户端的噪声比例
        for client_idx, sample_idx in dict_users.items():
            sample_idx = np.array(list(sample_idx))
            noisy_samples = np.sum(y_train_noisy_new[sample_idx] != y_train[sample_idx])  # 计算客户端的噪声样本数量
            total_samples = len(sample_idx)  # 计算客户端的总样本数量
            ratio = noisy_samples / total_samples  # 计算客户端的噪声比例
            noise_level_before[client_idx] = ratio  # 将噪声比例存储

        if args.correction:
            print("correcting",iteration)
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                loss = np.array(loss_accumulative_whole[sample_idx])
                local_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[idx] * args.relabel_ratio)]
                relabel_idx = list(
                    set(np.where(np.max(local_output, axis=1) > args.confidence_thres)[0]) & set(relabel_idx))
                # print("待修改样本索引：", relabel_idx)
                y_train_noisy_new = np.array(dataset_train.targets)
                # y_train_noisy = np.array(dataset_train.targets)
                y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]
                dataset_train.targets = y_train_noisy_new

            y_train_noisy_new = dataset_train.targets

            noise_level_after = np.zeros(len(real_noise_level))  # 存储每个客户端的噪声比例
            for client_idx, sample_idx in dict_users.items():
                sample_idx = np.array(list(sample_idx))
                noisy_samples = np.sum(y_train_noisy_new[sample_idx] != y_train[sample_idx])  # 计算客户端的噪声样本数量
                total_samples = len(sample_idx)  # 计算客户端的总样本数量
                ratio = noisy_samples / total_samples  # 计算客户端的噪声比例
                noise_level_after[client_idx] = ratio  # 将噪声比例存储

            # print("各个客户端初始噪声水平：", real_noise_level)
            # print("预训练阶段纠正前：各个客户端预测噪声水平：", estimated_noisy_level)
            # print("预训练阶段纠正前：各个客户端真实噪声水平：", noise_level_before)
            # print("预训练阶段纠正后：各个客户端真实噪声水平：", noise_level_after)

            x = range(len(real_noise_level))
            fig, ax = plt.subplots()
            ax.plot(x, real_noise_level, label='Initial Noise Level', linestyle='--')
            ax.plot(x, noise_level_before, label='Noise Level Before Correction')
            ax.plot(x, noise_level_after, label='Noise Level After Correction')
            ax.legend(loc=9)
            plt.xlabel('Client Index')
            plt.ylabel('Noise Level')
            plt.savefig("img/noise_level_"+str(iteration)+".jpg", bbox_inches='tight')
            plt.close()
        # 预处理阶段打印准确率
        acc_1 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)  # 测试全局模型在测试集上的准确率
        f_acc.write("preprocessing stage round %d, test acc  %.4f \n" % (iteration, acc_1))  # 将准确率写入文件
        print("preprocessing stage round {}, test acc {:.4f}".format(iteration, acc_1))  # 打印准确率
        f_acc.flush()  # 刷新文件
        accuracy.append(acc_1)

        # plot confusion matrix (take the first 5 clients for example)
        y_train_noisy_new = dataset_train.targets
        fig, axes = plt.subplots(1, 5, sharex=False, sharey=True, figsize=(18, 3), dpi=600)
        for i, ax in enumerate(axes):
            idx = list(dict_users[i])
            y_true = y_train[idx]
            y_noisy = y_train_noisy_new[idx]
            conf_matrix = confusion_matrix(y_true, y_noisy)
            im = ax.imshow(conf_matrix, cmap=plt.cm.hot_r)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title("Client {}".format(i + 1))  # 设置每个子图的标题为“客户端 i”
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.3)
            plt.colorbar(im, cax=cax)
        # 设置整个图的标签
        fig.text(0.5, 0.01, 'Corret Label', ha='center', fontsize=14)
        fig.text(0.1, 0.5, 'Noisy Label', va='center', rotation='vertical', fontsize=14)
        plt.savefig("img/noise_stat2.jpg", bbox_inches='tight')
        plt.close()

    # reset the beta,
    args.beta = 0

    # ---------------------------- second stage training -------------------------------
    if args.fine_tuning:
        # for i in range(len(estimated_noisy_level)):
        #     print("客户端",i,"噪声水平",estimated_noisy_level[i])
        selected_clean_idx = np.where(estimated_noisy_level <= args.clean_set_thres)[0]
        print("微调的干净客户端：", selected_clean_idx)
        prob = np.zeros(args.num_users)  # np.zeros(100)
        prob[selected_clean_idx] = 1 / len(selected_clean_idx)
        m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
        m = min(m, len(selected_clean_idx))
        netglob = copy.deepcopy(netglob)
        # add fl training
        for rnd in range(args.rounds1):
            w_locals, loss_locals = [], []
            idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
            print("微调客户端：", idxs_users)
            for idx in idxs_users:  # training over the subset
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                           w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
                w_locals.append(copy.deepcopy(w_local))  # store every updated model
                loss_locals.append(copy.deepcopy(loss_local))

            dict_len = [len(dict_users[idx]) for idx in idxs_users]
            # nl = noise_level_after[idxs_users]
            w_glob_fl = FedAvg(w_locals, dict_len)
            # w_glob_fl = FedAvgNL(w_locals, dict_len,nl)
            netglob.load_state_dict(copy.deepcopy(w_glob_fl))

            acc_2 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
            f_acc.write("fine tuning stage round %d, test acc  %.4f \n" % (rnd, acc_2))
            print("fine tuning stage round {}, test acc {:.4f}".format(rnd, acc_2))  # 打印准确率
            f_acc.flush()
            accuracy.append(acc_2)

        for client_idx, sample_idx in dict_users.items():
            sample_idx = np.array(list(sample_idx))
            noisy_samples = np.sum(y_train_noisy_new[sample_idx] != y_train[sample_idx])  # 计算客户端的噪声样本数量
            total_samples = len(sample_idx)  # 计算客户端的总样本数量
            ratio = noisy_samples / total_samples  # 计算客户端的噪声比例
            noise_level_before[client_idx] = ratio  # 将噪声比例存储

        if args.correction:
            relabel_idx_whole = []
            for idx in noisy_set:
                sample_idx = np.array(list(dict_users[idx]))
                dataset_client = Subset(dataset_train, sample_idx)
                loader = torch.utils.data.DataLoader(dataset=dataset_client, batch_size=100, shuffle=False)
                glob_output, _ = get_output(loader, netglob.to(args.device), args, False, criterion)
                y_predicted = np.argmax(glob_output, axis=1)
                relabel_idx = np.where(np.max(glob_output, axis=1) > args.confidence_thres)[0]
                y_train_noisy_new = np.array(dataset_train.targets)
                # y_train_noisy = np.array(dataset_train.targets)
                y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]
                dataset_train.targets = y_train_noisy_new

            for client_idx, sample_idx in dict_users.items():
                sample_idx = np.array(list(sample_idx))
                noisy_samples = np.sum(y_train_noisy_new[sample_idx] != y_train[sample_idx])  # 计算客户端的噪声样本数量
                total_samples = len(sample_idx)  # 计算客户端的总样本数量
                ratio = noisy_samples / total_samples  # 计算客户端的噪声比例
                noise_level_after[client_idx] = ratio  # 将噪声比例存储

            # print("各个客户端真实噪声水平：", real_noise_level)
            # print("微调阶段纠正前：各个客户端真实噪声水平：", noise_level_before)
            # print("微调阶段纠正后：各个客户端真实噪声水平：", noise_level_after)

            x = range(len(real_noise_level))
            x_ticks = [i for i in range(0, len(real_noise_level), 20)]
            fig, ax = plt.subplots()
            ax.plot(x, real_noise_level, label='Initial Noise Level',linestyle='--')
            ax.plot(x, noise_level_before, label='Noise Level Before Correction')
            ax.plot(x, noise_level_after, label='Noise Level After Correction')
            plt.xticks(x_ticks, x_ticks)
            ax.legend()
            # print("到这了")
            plt.xlabel('Client Index')
            plt.ylabel('Noise Level')
            plt.savefig("img/noise_level_finetuning.jpg", bbox_inches='tight')

            # plot confusion matrix (take the first 5 clients for example)
            y_train_noisy_new = dataset_train.targets
            fig, axes = plt.subplots(1, 5, sharex=False, sharey=True, figsize=(18, 3), dpi=600)
            for i, ax in enumerate(axes):
                idx = list(dict_users[i])
                y_true = y_train[idx]
                y_noisy = y_train_noisy_new[idx]
                conf_matrix = confusion_matrix(y_true, y_noisy)
                im = ax.imshow(conf_matrix, cmap=plt.cm.hot_r)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("Client {}".format(i + 1))  # 设置每个子图的标题为“客户端 i”
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.3)
                plt.colorbar(im, cax=cax)
            # 设置整个图的标签
            fig.text(0.5, 0.01, 'Corret Label', ha='center', fontsize=14)
            fig.text(0.1, 0.5, 'Noisy Label', va='center', rotation='vertical', fontsize=14)
            plt.savefig("img/noise_stat3.jpg", bbox_inches='tight')

    # ---------------------------- third stage training -------------------------------
    # third stage hyper-parameter initialization
    # args.beta  = 1 # FedProx
    m = max(int(args.frac2 * args.num_users), 1)  # num_select_clients
    prob = [1 / args.num_users for i in range(args.num_users)]
    # print("一般训练阶段选取可能：", prob)

    best_accuracy = 0.0
    max_rounds = int(args.rounds2*0.1)
    noimprove_rounds = 0
    for rnd in range(args.rounds2):
        w_locals, loss_locals = [], []
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p=prob)
        print("一般训练阶段选取客户端：", idxs_users)
        for idx in idxs_users:  # training over the subset
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w_local, loss_local = local.update_weights(net=copy.deepcopy(netglob).to(args.device), seed=args.seed,
                                                       w_g=netglob.to(args.device), epoch=args.local_ep, mu=0)
            w_locals.append(copy.deepcopy(w_local))  # store every updated model
            loss_locals.append(copy.deepcopy(loss_local))

        dict_len = [len(dict_users[idx]) for idx in idxs_users]
        # nl = noise_level_after[idxs_users] #用真实噪声水平去混合加权

        # w_glob_fl = FedAvg(w_locals, dict_len)
        # w_glob_fl = FedAvgNL(w_locals, dict_len, nl)
        # 降温聚合
        if rnd == 0:
            w_glob_fl = PerFedAvg(w_locals, w_locals, dict_len)
            w_last = w_locals
        else:
            w_glob_fl = PerFedAvg(w_locals,w_last, dict_len)

        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        acc_3 = globaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
        f_acc.write("third stage round %d, test acc  %.4f \n" % (rnd, acc_3))
        print("round {}, test acc {:.4f}".format(rnd, acc_3))  # 打印准确率
        f_acc.flush()
        accuracy.append(acc_3)
        accuracy3.append(acc_3)
        # 早停策略
        if acc_3 > best_accuracy :
            best_accuracy = acc_3
            bestnetglob = netglob
            torch.save(bestnetglob.state_dict(), 'weights/best.pt')
            noimprove_rounds = 0
        else:
            noimprove_rounds +=1
        if noimprove_rounds >= max_rounds:
            print("早停最佳准确率：",best_accuracy)
            f_acc.write("third stage round %d,early best test acc  %.4f \n" % (rnd, best_accuracy))
    torch.save(netglob.state_dict(), 'weights/last.pt')
    fig, ax = plt.subplots()
    epochs = range(1, args.iteration1+args.rounds1+args.rounds2 + 1)
    ax.plot(epochs, accuracy, linestyle='-')    # ax.legend()
    ax.plot(np.argmax(accuracy3)+args.iteration1+args.rounds1, np.max(accuracy3),'*')
    ax.vlines([args.iteration1+1, args.rounds1+1], 0, 1, linestyles='dashed', colors='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("img/model_acc.jpg")
    print("迭代到",np.argmax(accuracy3),"最佳准确率为：",np.max(accuracy3))
    f_acc.write("best test acc  %.4f \n" % np.max(accuracy))
    # print("早停最优准确率",early_accuracy)
    torch.cuda.empty_cache()