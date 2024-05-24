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
from util.local_training import finaltest
from util.fedavg import FedAvg
from util.fedavg_nl import FedAvgNL
from util.per_fed import PerFedAvg
from util.util import add_noise, lid_term, get_output
from util.dataset import get_dataset
from util.util import SCELoss
from model.build_model import build_model

np.set_printoptions(threshold=np.inf)

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
    # 数据划分
    dataset_train, dataset_test, dict_users = get_dataset(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 根据数据集获取类别编号与类别名称的映射关系
    if args.dataset == 'cifar10':
        class_labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    # if args.dataset == 'cifar100':

    if args.dataset == 'clothing1M':
        class_labels = {0: 'T-Shirt', 1: 'Shirt', 2: 'Knitwear', 3: 'Chiffon', 4: 'Sweater',
                        5: 'Hoodie', 6: 'Windbreaker', 7: 'Jacket', 8: 'Downcoat', 9: 'Suit',
                        10: 'Shawl', 11: 'Dress', 12: 'Vest', 13: 'Underwear'}
        class_names = ['T-Shirt', 'Shirt', 'Knitwear', 'Chiffon', 'Sweater', 'Hoodie'
            , 'Windbreaker', 'Jacket', 'Downcoat', 'Suit', 'Shawl', 'Dress',
                       'Vest', 'Underwear']

    # 模型建立
    netglob = build_model(args)
    # 指定模型权重的路径
    model_weights_path = "img/clothing1m_FedAvg/best.pt"
    # model_weights_path = "img/cifar_noise_0.6_dis0.7_1_mixup1/best.pt"
    state_dict = torch.load(model_weights_path)
    netglob.load_state_dict(state_dict)

    test_acc, true_labels, pred_labels = finaltest(copy.deepcopy(netglob).to(args.device), dataset_test, args)
    print("test acc {:.4f}".format(test_acc))  # 打印准确率
    # for true_label, pred_label in zip(true_labels, pred_labels):
    #     print("真实标签:", true_label, "| 预测标签:", pred_label)
    true_labels_en = [class_labels[label] for label in true_labels]
    pred_labels_en = [class_labels[label] for label in pred_labels]
    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    # 绘制混淆矩阵热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix acc:' + str(test_acc))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig('test/'+str(args.dataset)+'_confusion_matrix_acc_' + str(test_acc) + '.jpg', bbox_inches='tight')
    torch.cuda.empty_cache()
