
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import data_loader_1d
import resnet18_1d as models
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ------------------ 基本设置 ------------------
no_cuda = False
seed = 8
batch_size = 100
FFT = True
dataset = 'LWdatanew'
class_num = 4

cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

def forward_and_pseudo_update(x, model, optimizer, conf_thresh=0.0, lambda_cons=0.5):

    outputs = model(x)
    probs = F.softmax(outputs, dim=1)
    max_conf, pseudo_labels = probs.max(1)


    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    entropy_norm = entropy / np.log(probs.shape[1])  # 归一化熵 [0,1]

    weights = max_conf

    mask = max_conf >= conf_thresh
    selected_count = mask.sum().item()


    mean_prob = probs.mean(dim=0, keepdim=True)  # 平均分布
    cons_loss = ((probs - mean_prob) ** 2).sum(dim=1).mean()  # L2 一致性项

    if selected_count > 0:
        model.train()
        selected_outputs = outputs[mask]
        selected_labels = pseudo_labels[mask]
        selected_weights = weights[mask]


        loss_per_sample = F.cross_entropy(selected_outputs, selected_labels, reduction='none')
        weighted_loss = (loss_per_sample * selected_weights).mean()


        
        total_loss =lambda_cons * cons_loss+weighted_loss
        


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        model.eval()

    else:
        print("[DEBUG] No samples selected, skip update.")
        # 仍可基于一致性微调
        if lambda_cons > 0:
            model.train()
            total_loss = lambda_cons * cons_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            model.eval()

    return outputs



def test_target_TTA(model, test_loader, lr=1e-3, conf_thresh=1.0):
    """
    对每个 batch 使用高置信度伪标签更新模型
    打印每个 batch 的适应前后准确率，并统计适应前预测的正确/错误部分的平均置信度和熵
    """
    correct_total = 0
    total_loss = 0
    base_model = deepcopy(model)

    def compute_entropy(probs):
        return -(probs * torch.log(probs + 1e-8)).sum(dim=1)

    with torch.enable_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            # 每个 batch 用 base_model 初始化
            model = deepcopy(base_model)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            if cuda:
                data, label = data.cuda(), label.cuda()


            with torch.no_grad():
                outputs_before = model(data)
                probs_before = F.softmax(outputs_before, dim=1)
                entropy_before = compute_entropy(probs_before)
                max_conf_before = probs_before.max(1)[0]

                pred_before = outputs_before.argmax(1)
                acc_before = (pred_before == label).float().mean().item()


            outputs = forward_and_pseudo_update(data, model, optimizer, conf_thresh)


            with torch.no_grad():
                outputs_after = model(data)
                pred_after = outputs_after.argmax(1)
                acc_after = (pred_after == label).float().mean().item()


            # ------------------ 累计统计 ------------------
            correct_total += (pred_after == label).sum().item()
            total_loss += F.cross_entropy(outputs_after, label, reduction='sum').item()

    acc = correct_total / len(test_loader.dataset)
   
    return acc


# ------------------ 普通测试（不更新模型） ------------------
def test_target_plain(model, test_loader):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, label in test_loader:
            if cuda:
                data, label = data.cuda(), label.cuda()

            outputs = model(data)
            total_loss += F.cross_entropy(outputs, label, reduction='sum').item()
            pred = outputs.argmax(1)
            correct += (pred == label).sum().item()
    acc = correct / len(test_loader.dataset)
    # print(f"Plain Test - Loss: {total_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc * 100:.2f}%)")
    return acc


# ------------------ 主流程 ------------------
if __name__ == "__main__":
    Source = np.array(
        [
            [0, 1, 4],
            [0, 1, 5],
            [0, 2, 4],
            [0, 3, 4],
            [0, 3, 5],
            [1, 2, 4],
            [1, 2, 5],
            [1, 3, 5],
            [2, 3, 4],
            [3, 4, 5]

        ])

    Target = np.array(
        [
            [2, 3, 5],
            [2, 3, 4],
            [1, 3, 5],
            [1, 2, 5],
            [1, 2, 4],
            [0, 3, 5],
            [0, 3, 4],
            [0, 2, 4],
            [0, 1, 5],
            [0, 1, 2],

        ]
    )

    for task_index in range(10):
        sourcelist = Source[task_index]
        targetlist = Target[task_index]

        root_path = f'D:\\ZHAOCHAO\\{dataset}{class_num}.mat'
        kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

        tgt_loader = data_loader_1d.load_testing_vib(root_path, targetlist, FFT, class_num, batch_size, kwargs)

        model = models.CNN_1D(num_classes=class_num)
        if cuda: model.cuda()

        # ------------------ 加载训练好的模型 ------------------

        trained_model_path = f'D:\\ZHAOCHAO\Paper-20\\re2\\pretrained_model\\PEM_task{task_index}.pth'
        
        
        model.load_state_dict(torch.load(trained_model_path))

        # ------------------ Plain Test ------------------
        acc_plain = test_target_plain(model, tgt_loader)
        

        # ------------------ Pseudo-label TTA ------------------
        acc_TTA = test_target_TTA(model, tgt_loader, lr=0.002, conf_thresh=0.0)

        print(acc_TTA*100)

