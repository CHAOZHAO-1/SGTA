#author:zhaochao time:2021/5/26
import numpy as np
import  torch as t
import random
import torch.nn as nn
import torch.nn.functional as F


def cal_reconstruction_loss( x, x_rec):
    # return (x_rec-x).pow(2).sum()/x.shape[0]
    return (x_rec - x).pow(2).mean()

def cal_reduce_redundancy_loss(fm_vec, fh_vec):
    '''
    zz = torch.load('fm_fh_tensor.pt',map_location=torch.device('cpu') )
    fm_vec = zz[0]
    fh_vec = zz[1]
    '''
    lbd = 1
    B = fm_vec.shape[0]
    D = fm_vec.shape[1]
    # debug
    # torch.save([fm_vec, fh_vec],'fm_fh_tensor.pt')
    # 注意这里，原来是在dim=1上标准化，这个是错误的，这里一列才是一个vector，所以应该是dim=0标准化
    fm_vec = F.normalize(fm_vec, p=2, dim=0)  # (B,D)
    fh_vec = F.normalize(fh_vec, p=2, dim=0)  # (B,D)
    sim_fm_vec = t.matmul(fm_vec.T, fm_vec)  # (D,D)
    sim_fh_vec = t.matmul(fh_vec.T, fh_vec)  # (D,D)
    # 经过normalize之后，上边两个矩阵的对角线本身就是1（不同于Barlow Twins, 这里是两个相同向量的内积）

    # E = t.eye(D).to(self.device)

    E = t.eye(D).cuda()

    loss_fm = ((1 - E) * sim_fm_vec).pow(2).sum() / t.sum(1 - E)
    loss_fh = ((1 - E) * sim_fh_vec).pow(2).sum() / t.sum(1 - E)

    loss_fmh = t.matmul(fh_vec.T, fm_vec).div(B).pow(2).mean()
    # loss = loss_fmh

    loss = loss_fm + loss_fh + loss_fmh
    # loss =   loss_fm + loss_fh

    return loss


def cal_causal_aggregation_loss(fm_vec, fh_vec, labels, domain_labels):
    B = fm_vec.shape[0]
    D = fm_vec.shape[1]

    fm_vec = F.normalize(fm_vec, p=2, dim=1)  # (B,D)
    fh_vec = F.normalize(fh_vec, p=2, dim=1)  # (B,D)

    labels = labels.contiguous().view(-1, 1)
    # mask_fh = t.eq(labels, labels.T).float().to(self.device)  # (B,B)

    mask_fh = t.eq(labels, labels.T).float().cuda()


    sim_fh_vec = t.matmul(fh_vec, fh_vec.T) / D  # (B,B)
    loss_fh = -(mask_fh * sim_fh_vec).sum() / t.sum(mask_fh) + ((1 - mask_fh) * sim_fh_vec).sum() / t.sum(
        1 - mask_fh)
    # loss_fh = -(mask_fh*sim_fh_vec).sum() + ((1-mask_fh)*sim_fh_vec).sum() #似乎会带来不稳定（在MFS表现上）

    domain_labels = domain_labels.contiguous().view(-1, 1)
    mask_fm = t.eq(domain_labels, domain_labels.T).float().cuda(0)
    sim_fm_vec = t.matmul(fm_vec, fm_vec.T) / D  # (B,B)
    loss_fm = -(mask_fm * sim_fm_vec).sum() / t.sum(mask_fm) + ((1 - mask_fm) * sim_fm_vec).sum() / t.sum(
        1 - mask_fm)
    # loss_fm = -(mask_fm*sim_fm_vec).sum() + ((1-mask_fm)*sim_fm_vec).sum()

    loss = loss_fm + loss_fh

    return loss


class Center_loss(nn.Module):
    def __init__(self,src_class):
        super(Center_loss, self).__init__()

        self.n_class=src_class
        self.MSELoss = nn.MSELoss()  # (x-y)^2
        self.MSELoss = self.MSELoss.cuda()



    def forward(self, s_feature,s_labels):


        n, d = s_feature.shape

        # get labels


        # image number in each class
        ones = t.ones_like(s_labels, dtype=t.float)
        zeros = t.zeros(self.n_class)

        zeros = zeros.cuda()

        s_n_classes = zeros.scatter_add(0, s_labels, ones)


        # image number cannot be 0, when calculating centroids
        ones = t.ones_like(s_n_classes)
        s_n_classes = t.max(s_n_classes, ones)


        # calculating centroids, sum and divide
        zeros = t.zeros(self.n_class, d)

        zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, t.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)

        s_centroid = t.div(s_sum_feature, s_n_classes.view(self.n_class, 1))


        # calculating inter distance

        temp = t.zeros((n, d)).cuda()

        for i in range(n):
            temp[i] = s_centroid[s_labels[i]]

       #
        # intra_loss = t.norm(temp-s_feature, p=1, dim=0).sum()
        # intra_loss = intra_loss / (d * n)

        #### way 1:
        intra_loss = self.MSELoss(temp, s_feature)



        return intra_loss

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)  # cpu
    t.cuda.manual_seed_all(seed)  # 并行gpu
    t.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    t.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速

def log(name1,name2,Train_Loss_list,Train_Accuracy_list,Test_Loss_list,Test_Accuracy_list,Train_Time):
    f = open('./'+name1+'/'+name2+'.txt', 'w')

    f.write('train_loss:')
    f.write(str(Train_Loss_list))
    f.write('\r\n')
    f.write('train_acc:')
    f.write(str(Train_Accuracy_list))
    f.write('\r\n')
    f.write('test_loss:')
    f.write(str(Test_Loss_list))
    f.write('\r\n')
    f.write('test_acc:')
    f.write(str(Test_Accuracy_list))
    f.write('\r\n')
    f.write('train_time:')
    f.write(str(Train_Time))
    f.close()


def log1(name1,name2,Train_Loss_list,Train_Accuracy_list,Test_Loss_list,Test_Accuracy_list,Train_Time,Train_Time1):
    f = open('./'+name1+'/'+name2+'.txt', 'w')

    f.write('train_loss:')
    f.write(str(Train_Loss_list))
    f.write('\r\n')
    f.write('train_acc:')
    f.write(str(Train_Accuracy_list))
    f.write('\r\n')
    f.write('test_loss:')
    f.write(str(Test_Loss_list))
    f.write('\r\n')
    f.write('test_acc:')
    f.write(str(Test_Accuracy_list))
    f.write('\r\n')
    f.write('train_time:')
    f.write(str(Train_Time))
    f.write('\r\n')
    f.write('train_time:')
    f.write(str(Train_Time1))
    f.close()

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (t.device('cuda')
                  if features.is_cuda
                  else t.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = t.eye(batch_size, dtype=t.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = t.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = t.cat(t.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = t.div(
            t.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = t.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = t.scatter(
            t.ones_like(mask),
            1,
            t.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = t.exp(logits) * logits_mask
        log_prob = logits - t.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss