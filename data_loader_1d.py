from torchvision import datasets, transforms
import torch
import numpy as np
from scipy.fftpack import fft
import scipy.io as scio
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import zscore




def min_max(Z):
    Zmin = Z.min(axis=1)
    Z = np.log(Z - Zmin.reshape(-1, 1) + 1)
    return Z


def load_training_vib(root_path, domainlist, fft1, class_num, batch_size, kwargs):

    data = scio.loadmat(root_path)

    # --------- 统一读取特征 ----------
    all_features = []
    all_labels = []

    for idx, domain_id in enumerate(domainlist):
        # 提取每个域的原始信号
        raw_data = data['load' + str(domain_id)]

        # 特征处理（FFT 或原始信号）
        if fft1:
            domain_fea = zscore(min_max(abs(fft(raw_data))[:, 0:1024]), axis=0)

        else:
            domain_fea = zscore(raw_data.T, axis=0).T

        # 每个类别 100 个样本，构造标签 (类别, 域)
        num_samples = 100 * class_num
        label = torch.zeros((num_samples, 2))
        for i in range(num_samples):
            label[i, 0] = i // 100  # 类别标签
            label[i, 1] = idx  # 当前是第几个源域

        all_features.append(domain_fea)
        all_labels.append(label)

    # --------- 合并所有域 ----------
    train_fea = np.vstack(all_features)
    train_label = torch.cat(all_labels, dim=0)

    print(f"[load_training_vib] 加载 {len(domainlist)} 个源域，共 {train_fea.shape[0]} 条样本")

    # --------- 构建 DataLoader ----------
    train_label = train_label.long()
    train_fea = torch.tensor(train_fea, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(train_fea, train_label)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    return loader


def load_testing_vib(root_path, domainlist, fft1, class_num, batch_size, kwargs):
    data = scio.loadmat(root_path)

    all_features = []
    all_labels = []

    for idx, domain_id in enumerate(domainlist):
        # ====== 提取原始数据 ======
        raw_data = data['load' + str(domain_id)]

        # ====== 特征处理 ======
        if fft1:
            domain_fea = zscore(min_max(abs(fft(raw_data))[:, 0:1024]), axis=0)
        else:
            domain_fea = zscore(raw_data.T, axis=1).T

        # ====== 生成标签 (只包含类别标签) ======
        num_samples = 100 * class_num
        label = torch.zeros((num_samples,))
        for i in range(num_samples):
            label[i] = i // 100  # 每100个样本一个类别

        all_features.append(domain_fea)
        all_labels.append(label)

    # ====== 合并所有目标域数据 ======
    test_fea = np.vstack(all_features)
    test_label = torch.cat(all_labels, dim=0).long()

    print(f"[load_testing_vib] 加载 {len(domainlist)} 个目标域，共 {test_fea.shape[0]} 条样本")

    # ====== 转成 DataLoader ======
    test_fea = torch.tensor(test_fea, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(test_fea, test_label)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return loader




