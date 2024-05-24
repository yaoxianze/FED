# python version 3.7.1
# -*- coding: utf-8 -*-

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    # 预处理阶段迭代次数
    parser.add_argument('--rounds1', type=int, default=200, help="rounds of training in fine_tuning stage")
    # 微调阶段迭代轮数
    parser.add_argument('--rounds2', type=int, default=400, help="rounds of training in usual training stage")
    # 常规训练阶段迭代轮数
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs")
    # 本地训练epoch数
    parser.add_argument('--frac1', type=float, default=0.01, help="fration of selected clients in preprocessing stage")
    # 预处理阶段选择客户端的几率
    parser.add_argument('--frac2', type=float, default=0.1,
                        help="fration of selected clients in fine-tuning and usual training stage")
    # 微调和常规训练阶段选择客户端几率
    parser.add_argument('--num_users', type=int, default=100, help="number of uses: K")
    # 客户端个数
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B") #64
    # 本地客户端训练batch大小
    parser.add_argument('--lr', type=float, default=0.02, help="learning rate")
    # 学习率
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum, default 0.5")
    # 动量
    parser.add_argument('--beta', type=float, default=5, help="coefficient for local proximal")
    # 损失函数正则化中的客户端近端系数

    # noise arguments
    parser.add_argument('--LID_k', type=int, default=20, help="lid")
    # 估计LID分数的相邻样本数
    parser.add_argument('--level_n_system', type=float, default=0.6, help="fraction of noisy clients")
    # 带噪客户端比例
    parser.add_argument('--level_n_lowerb', type=float, default=0.5, help="lower bound of noise level")
    # 局部噪音水平下限

    # correction
    parser.add_argument('--relabel_ratio', type=float, default=0.5,
                        help="proportion of relabeled samples among selected noisy samples")
    # 纠正标签概率
    parser.add_argument('--confidence_thres', type=float, default=0.5,
                        help="threshold of model's confidence on each sample")
    # 各个样本标签纠正阈值
    parser.add_argument('--clean_set_thres', type=float, default=0.1,
                        help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")
    # 微调阶段干净客户端噪声水平最低阈值

    # ablation study
    parser.add_argument('--fine_tuning', action='store_false', help='whether to include fine-tuning stage')
    # 是否包括微调阶段
    parser.add_argument('--correction', action='store_false', help='whether to correct noisy labels')
    # 是否进行噪声标签纠正

    # other arguments
    # parser.add_argument('--server', type=str, default='none', help="type of server")
    parser.add_argument('--model', type=str, default='resnet18', help="model name")
    # 客户端模型
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    # 数据集
    parser.add_argument('--pretrained', action='store_true', help="whether to use pre-trained model")
    # 是否预训练
    parser.add_argument('--iid', action='store_true', help="i.i.d. or non-i.i.d.")
    # 数据划分方式
    parser.add_argument('--non_iid_prob_class', type=float, default=0.7, help="non iid sampling prob for class")
    # 各个类的non iid采样概率
    parser.add_argument('--alpha_dirichlet', type=float, default=10)
    # 迪利克雷分布参数
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    # 类别个数
    parser.add_argument('--seed', type=int, default=133, help="random seed, default: 1") # 3405
    # 随机种子设置
    parser.add_argument('--mixup', action='store_false')
    # 是否使用Mixup
    parser.add_argument('--alpha', type=float, default=1.3, help="0.1,1,5")
    # Mixup参数
    return parser.parse_args()
