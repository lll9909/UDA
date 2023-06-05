import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss
def get_multi_dis(target, center, temperature=0.5):
    # attention_distribution = []
    attention_distribution = torch.zeros(center.size(0), target.size(0))
    for i in range(center.size(0)):
        attention_score = torch.exp(torch.cosine_similarity(target, center[i].view(1, -1).repeat(target.size(0), 1)) / temperature)  # 计算每一个元素与给定元素的余弦相似度
        # attention_distribution.append(attention_score.tolist())
        attention_distribution[i, :] = attention_score
    # attention_distribution = torch.Tensor(attention_distribution)
    multi_dis = attention_distribution / torch.sum(attention_distribution, 0)
    return multi_dis.t()

prev_parms = []
def w_loss(net, iter, weight):
    sw_loss = torch.tensor(0.0).to(torch.float64).cuda()
    global prev_parms
    if iter == 0:
        iter += 1
        for name, parms in net.named_parameters():
            par = torch.zeros(parms.shape).cuda()
            par = par.copy_(parms)
            prev_parms.append(par.to(torch.float64))
        return sw_loss
    elif iter > 0:
        n = 0
        reg = torch.tensor(0.0).to(torch.float64)
        for name, parms in net.named_parameters():
            parms = parms.to(torch.float64)
            reg = (parms.to(torch.float64) - prev_parms[n])**2
            sw_loss += weight[0,n]*reg.sum()
            n += 1
        prev_parms = []
        for name, parms in net.named_parameters():
            par = torch.zeros(parms.shape).cuda()
            par = par.copy_(parms)
            prev_parms.append(par.to(torch.float64))
        return sw_loss
def sup_loss(representations,label,T=0.5):
    n = label.shape[0]
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    #这步得到它的label矩阵，相同label的位置为1
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

    #这步得到它的不同类的矩阵，不同类的位置为1
    mask_no_sim = torch.ones_like(mask) - mask
    mask_no_sim = mask_no_sim.cuda()
    #这步产生一个对角线全为0的，其他位置为1的矩阵
    mask_dui_jiao_0 = torch.ones(n ,n) - torch.eye(n, n )
    mask_dui_jiao_0 = mask_dui_jiao_0.cuda()
    #这步给相似度矩阵求exp,并且除以温度参数T
    similarity_matrix = torch.exp(similarity_matrix/T)

    #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
    similarity_matrix = similarity_matrix*mask_dui_jiao_0


    #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
    sim = mask*similarity_matrix


    #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
    no_sim = similarity_matrix - sim


    #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
    no_sim_sum = torch.sum(no_sim, dim=1)

    '''
    将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
    至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
    每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
    '''
    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)


    '''
    由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
    全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
    '''
    loss = mask_no_sim + loss + torch.eye(n, n).cuda()


    #接下来就是算一个批次中的loss了
    loss = -torch.log(loss)  #求-log
    # loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n

    # print(loss)  #0.9821
    #最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
    loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))
    return loss
