
import torch
import torch.nn.functional as F

from torch.autograd import Variable
import os
import math
import data_loader_1d
import resnet18_1d as models

import  time
import numpy as np
import  random
from  utils import  *



os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings



momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4


def train_PEM(model, rho=0.05):
    """

    """
    src_iter = iter(src_loader)
    Train_Loss_list = []
    Train_Accuracy_list = []
    Test_Loss_list = []
    Test_Accuracy_list = []

    start = time.time()

    best_test_acc=0

    for i in range(1, iteration + 1):

        model.train()

        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

        try:
            src_data, src_label = src_iter.next()
        except Exception:
            src_iter = iter(src_loader)
            src_data, src_label = next(src_iter)

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()

        src_label = src_label[:,0]


        optimizer.zero_grad()
        logits = model(src_data)
        cls_loss = F.nll_loss(F.log_softmax(logits, dim=1), src_label)
        cls_loss.backward()


        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += (p.grad.data ** 2).sum()
        grad_norm = grad_norm.sqrt()


        e_w_list = []
        for p in model.parameters():
            if p.grad is None:
                e_w_list.append(None)
                continue
            e_w = rho * p.grad / (grad_norm + 1e-12)
            p.data.add_(e_w)
            e_w_list.append(e_w)


        optimizer.zero_grad()
        logits_perturbed = model(src_data)
        loss_perturbed = F.nll_loss(F.log_softmax(logits_perturbed, dim=1), src_label)
        loss_perturbed.backward()


        for p, e_w in zip(model.parameters(), e_w_list):
            if e_w is not None:
                p.data.sub_(e_w)

        optimizer.step()

        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss_perturbed.item(), cls_loss.item()))

        if i % (log_interval * 10) == 0:
            train_correct, train_loss = test_source(model, src_loader)
            test_correct, test_loss = test_target(model, tgt_test_loader)
            Train_Accuracy_list.append(train_correct.cpu().numpy() / len(src_loader.dataset))
            Train_Loss_list.append(train_loss)
            Test_Accuracy_list.append(test_correct.cpu().numpy() / len(tgt_test_loader.dataset))
            Test_Loss_list.append(test_loss)





def test_source(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            tgt_test_label=tgt_test_label[:,0]
            # print(tgt_test_data)
            tgt_pred= model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()


    print('\n Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
    return correct,test_loss


def test_target(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)
            # print(tgt_test_data)
            tgt_pred= model(tgt_test_data)
            test_loss += F.nll_loss(F.log_softmax(tgt_pred, dim=1), tgt_test_label,
                                    reduction='sum').item()  # sum up batch loss
            pred = tgt_pred.data.max(1)[1]  # get the index of the max log-probability

            correct += pred.eq(tgt_test_label.data.view_as(pred)).cpu().sum()


    print('\n Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
    return correct,test_loss


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format( total_num, trainable_num))


if __name__ == '__main__':

    # setup_seed(seed)
    iteration = 8000

    batch_size =256

    lr=0.005

    FFT = True

    dataset = 'LWdatanew'

    class_num = 4


    Source=np.array(
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

    for taskindex in range(10):
        sourcelist = Source[taskindex]
        targetlist = Target[taskindex]

        for repeat in range(1):

            root_path = 'D:\\ZHAOCHAO\\' + dataset + str(class_num) + '.mat'

            cuda = not no_cuda and torch.cuda.is_available()
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

            kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

            src_loader = data_loader_1d.load_training_vib(root_path, sourcelist, FFT, class_num, batch_size, kwargs)

            tgt_test_loader = data_loader_1d.load_testing_vib(root_path, targetlist, FFT, class_num, batch_size, kwargs)

            src_dataset_len = len(src_loader.dataset)

            src_loader_len = len(src_loader)

            model = models.CNN_1D(num_classes=class_num)
            # get_parameter_number(model) 计算模型训练参数个数
            print(model)

            if cuda:
                model.cuda()

            train_PEM(model)






