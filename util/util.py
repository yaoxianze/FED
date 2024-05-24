import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist


def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)
    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c >= 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.num_classes, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    fig = plt.Figure

    return (y_train_noisy, gamma_s, real_noise_level)


def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if latent == False:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1] + eps))) + eps)  # 定义函数，用于计算局部内在维度（Local Intrinsic Dimensionality）
    distances = cdist(X, batch) # 计算数据点和批次中每个点之间的距离

    # 获取最近的k个邻居
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # 排序后的距离矩阵
    sort_distances = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=sort_distances)
    return lids




class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels.long())  # zzy修改.long()

        # RCE
        pred = F.softmax(pred, dim=1)  # softmax函数将原始分数转换为标准化的概率分布，使得概率之和为1
        pred = torch.clamp(pred, min=1e-7, max=1.0)  # clamp将张量pred中的所有元素裁剪到指定的取值范围内
        label_one_hot = torch.nn.functional.one_hot(labels.to(torch.int64), self.num_classes).float().to(
            self.device)  # zzy修改to(torch.int64)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


def calculate_metrics(true_labels, predicted_labels):
    TP = FP = TN = FN = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        # 如果样本被正确地标记为标签噪声
        if true_label == 0:
            if predicted_label == 0:
                TP += 1  # 真正例
            else:
                FP += 1  # 假负例
        # 如果样本被正确地标记为非标签噪声
        else:
            if predicted_label == 1:
                TN += 1  # 真负例
            else:
                FN += 1  # 假正例

    return TP, FP, TN, FN