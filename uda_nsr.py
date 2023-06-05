import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
import time
import csv
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Dataset
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
import scipy.io as sio

from data_load import mnist, svhn, usps
from multi_scale_block import *
import network, loss

import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_with_labels(lowDWeights, labels, epoch):

    plt.cla() #clear当前活动的坐标轴
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1] #把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 3));
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        # plt.text(x, y, str(s),color=c,fontdict={'weight': 'bold', 'size': 9}) #在指定位置放置文本
    plt.xlim(X.min(), X.max());
    plt.ylim(Y.min(), Y.max());
    plt.title('t-SNE Visual')
    plt.savefig('visual{}.png'.format(epoch), dpi=600)

    plt.show()
    plt.pause(0.01)

class MyDataset(Dataset):
    def __init__(self, data, labels, idx):
        self.data = data
        self.labels = labels
        self.idx = idx # 我的例子中label是一样的，如果你的不同，再增加一个即可

    def __getitem__(self, index):
        data, labels, idx = self.data[index], self.labels[index], self.idx[index]
        data1 = data
        data2 = trans(data)
        return (data1, data2), labels, idx

    def __len__(self):
        return len(self.data) # 我的例子中len(self.data1) = len(self.data2)
def recycle(arr):
    kk = arr.size(1)
    k = int((2*np.random.random()-1)*kk)
    result = torch.roll(arr, k, 1)
    return result
class Recycle(nn.Module):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, signal):
        if random.uniform(0, 1) < self.p:  # 按概率执行该操作
            k_re = int((2*np.random.random()-1)*signal.size(1))
            signal = torch.roll(signal, k_re, 1)
            return signal
        else:
            return signal
class Mask(nn.Module):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, signal, mask_ratio=0.1):
        if random.uniform(0, 1) < self.p:  # 按概率执行该操作
            mask_len = int(signal.size(1)*mask_ratio)
            mask_start = int(np.random.random()*signal.size(1)) - mask_len
            signal[:, mask_start:mask_start+mask_len] = torch.tensor(0.0)
            return signal
        else:
            return signal
class Mirror(nn.Module):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, signal):
        if random.uniform(0, 1) < self.p:  # 按概率执行该操作
            signal = signal.flip(dims=[1])
            return signal
        else:
            return signal
class RandomNoise(nn.Module):
    def __init__(self, snr=10, p=0.5):
        self.p = p
        self.snr = snr
    def __call__(self, signal):
        if random.uniform(0, 1) < self.p:  # 按概率执行该操作
            batch_size, len_s = signal.shape
            Ps = torch.sum(torch.pow(signal, 2)) / len_s
            Pn = Ps / (np.power(10, self.snr / 10))
            noise = torch.randn(len_s) * np.sqrt(Pn)
            return signal + noise
        else:
            return signal

trans = nn.Sequential(Recycle(), RandomNoise())

def sim_cos(vec1, vec2):
    sim = torch.tensor(0.0)
    if len(vec1) != len(vec2):
        print("张量维度不一样，不符合要求！")
    else:
        sim = (vec1*vec2).sum()/(torch.sqrt((vec1*vec1).sum())*torch.sqrt((vec2*vec2).sum())+1e-12)
    return sim


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def digit_load(args, num=1):
    train_bs = args.batch_size // num
    if args.dset == 's2m':
        train_source = svhn.SVHN('./data/svhn/', split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        test_source = svhn.SVHN('./data/svhn/', split='test', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))  
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))      
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
    elif args.dset == 'che':
        train_source = ImageFolder('./data/lll2/train', transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    #transforms.RandomResizedCrop(224),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        test_source = ImageFolder('./data/lll2/valid', transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        train_target = ImageFolder('./data/lll2/test601', transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        test_target = ImageFolder('./data/lll2/test601', transform=transforms.Compose([
                    transforms.Resize((224, 224)),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
    elif args.dset == 'shice':
        data = sio.loadmat('3lisan.mat')
        all_data = data['all_data']
        all_label = data['all_label']

        all_num = np.size(all_data, 0)
        select_index = torch.randperm(all_num)
        all_data = all_data[select_index, :]
        all_label = all_label[select_index, :]
        train_ratio = 7/10
        train_num = int(all_num*train_ratio)

        train_data = all_data[0:train_num, :]
        train_label = all_label[0:train_num, :]
        num_train_instances = len(train_data)
        train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
        train_label = torch.from_numpy(train_label).type(torch.LongTensor)
        train_data = train_data.view(num_train_instances, 1, -1)
        train_label = np.squeeze(train_label.view(num_train_instances, 1))
        train_index = torch.arange(num_train_instances)
        train_source = MyDataset(train_data, train_label, train_index)

        test_data = all_data[train_num:, :]
        test_label = all_label[train_num:, :]

        num_test_instances = len(test_data)
        test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
        test_label = torch.from_numpy(test_label).type(torch.LongTensor)
        test_data = test_data.view(num_test_instances, 1, -1)
        test_label = np.squeeze(test_label.view(num_test_instances, 1))
        test_index = torch.arange(num_test_instances)
        test_source = MyDataset(test_data, test_label, test_index)

        data2 = sio.loadmat('3lisandB5.mat')
        all_data2 = data2['all_data_noise']
        all_label2 = data2['all_label']
        all_num = np.size(all_data2, 0)
        select_index = torch.randperm(all_num)
        all_data2 = all_data2[select_index, :]
        all_label2 = all_label2[select_index, :]
        train_ratio = 10/10
        train_num = int(all_num*train_ratio)
        #
        train_data2 = all_data2[0:train_num, :]
        train_label2 = all_label2[0:train_num, :]


        num_train_instances = len(train_data2)
        train_data2 = torch.from_numpy(train_data2).type(torch.FloatTensor)
        train_label2 = torch.from_numpy(train_label2).type(torch.LongTensor)
        train_data2 = train_data2.view(num_train_instances, 1, -1)
        train_label2 = np.squeeze(train_label2.view(num_train_instances, 1))
        train_index = torch.arange(num_train_instances)
        train_target = MyDataset(train_data2, train_label2, train_index)

        test_data2 = train_data2
        test_label2 = train_label2
        test_index = train_index
        # test_data2 = all_data2[train_num:, :]
        # test_label2 = all_label2[train_num:, :]
        # num_test_instances = len(test_data2)
        # test_data2 = torch.from_numpy(test_data2).type(torch.FloatTensor)
        # test_label2 = torch.from_numpy(test_label2).type(torch.LongTensor)
        # test_data2 = test_data2.view(num_test_instances, 1, -1)
        # test_label2 = np.squeeze(test_label2.view(num_test_instances, 1))
        # test_index = torch.arange(num_test_instances)
        test_target = MyDataset(test_data2, test_label2, test_index)

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["source_w"] = DataLoader(train_source, batch_size=1, shuffle=True,
        num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)
    #concat_test = ConcatDataset([train_source,train_target])
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=False,
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs, shuffle=False,
        num_workers=args.worker, drop_last=False)
    return dset_loaders

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs, _ = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == torch.squeeze(all_label)).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent

def train_source(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()
    elif args.dset == 'che':
        netF = network.Res50().cuda()
    elif args.dset == 'shice':
        netF = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=args.class_num).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   

    #optimizer = optim.SGD(param_group)
    optimizer = optim.Adam(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0.0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    #interval_iter = max_iter // 10
    interval_iter = len(dset_loaders["source_tr"])
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    while iter_num < max_iter:
        try:
            (inputs_source,_), labels_source, _ = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            (inputs_source,_), labels_source, _ = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        # print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        feature_source = netB(netF(inputs_source))
        sup_loss = loss.sup_loss(feature_source,labels_source)
        # optimizer.zero_grad()
        # sup_loss.backward()
        # optimizer.step()

        outputs_source = netC(netB(netF(inputs_source)))
    
        classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
        all_loss = classifier_loss + 0.5*sup_loss
        train_loss.append(all_loss.item())
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            with torch.no_grad():
                try:
                    (val_source,_), labels_val, _ = val_source.next()
                except:
                    val_source = iter(dset_loaders["source_te"])
                    (val_source,_), labels_val, _ = val_source.next()
                val_source,labels_val = val_source.cuda(),labels_val.cuda()
                feature_source2 = netB(netF(val_source))
                sup_loss2 = loss.sup_loss(feature_source2,labels_val)
                outputs_source2 = netC(netB(netF(val_source)))
                classifier_loss2 = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
                all_loss2 = classifier_loss2
                val_loss.append(all_loss2.item())

            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netB, netC)
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
            train_acc.append(acc_s_tr)
            val_acc.append(acc_s_te)
            log_str = 'Task-sourece-train/test: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%;train_loss{}/val_loss{}'.format(args.dset, iter_num, max_iter, acc_s_tr, acc_s_te, all_loss, all_loss2)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            with open(args.output_dir + '/train.csv', 'w') as outf:
                writer = csv.writer(outf, dialect='excel')
                writer.writerow(tuple(train_loss))
                writer.writerow(tuple(val_loss))
                writer.writerow(tuple(train_acc))
                writer.writerow(tuple(val_acc))

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
                torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
                torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
                torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))
            print('Iter:{}/{}; acc_source_test:{}'.format(iter_num, max_iter, acc_init)+'\n')
            netF.train()
            netB.train()
            netC.train()

    torch.save(netF.state_dict(), osp.join(args.output_dir, "source_F_f.pt"))
    torch.save(netB.state_dict(), osp.join(args.output_dir, "source_B_f.pt"))
    torch.save(netC.state_dict(), osp.join(args.output_dir, "source_C_f.pt"))


    return netF, netB, netC

def test_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()
    elif args.dset == 'che':
        netF = network.Res50().cuda()
    elif args.dset == 'shice':
        netF = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=args.class_num).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = 'Task-sourece-target: {}, Accuracy = {:.2f}%'.format(args.dset, acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

from utils import set_requires_grad, loop_iterable
def project_target(args):
    dset_loaders = digit_load(args, num=1)
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'che':
        netF = network.Res50().cuda()
    elif args.dset == 'shice':
        netF = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=args.class_num).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    # netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netP = network.MLP(in_features=256, hidden_features=1024, projection_features=256).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'
    netFB = nn.Sequential(netF, netB)
    netFB.eval()
    netF.eval()
    netB.eval()
    set_requires_grad(netFB, requires_grad=False)
    for k, v in netF.named_parameters():
         v.requires_grad = False
    for k, v in netB.named_parameters():
         v.requires_grad = False


    source_loader = dset_loaders["source_tr"]
    optimizer = torch.optim.Adam([{'params': netP.parameters()}], lr=0.003)

    center0 = []
    center1 = []
    center2 = []
    # center3 = []
    for step, ((x_i, x_j), labels, _) in enumerate(source_loader):
        x_i = x_i.to(args.device)
        with torch.no_grad():
            feature = netP(netFB(x_i))
        for i in range(len(labels)):
            if labels[i] == 0:
                center0.append(feature[i].tolist())
            elif labels[i] == 1:
                center1.append(feature[i].tolist())
            elif labels[i] == 2:
                center2.append(feature[i].tolist())
            # elif labels[i] == 3:
            #     center3.append(feature[i].tolist())
    center0 = torch.tensor(center0).mean(dim=0)
    center1 = torch.tensor(center1).mean(dim=0)
    center2 = torch.tensor(center2).mean(dim=0)
    # center3 = torch.tensor(center3).mean(dim=0)
    center = torch.stack([center0, center1, center2], dim=0).to(args.device)
    for epoch in range(1, args.proepoch):
        center0 = []
        center1 = []
        center2 = []
        # center3 = []
        for step, ((x_i, x_j), labels, _) in enumerate(source_loader):
            """加载数据至GPU"""
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)
            labels = labels.to(args.device)
            """计算在线网络和目标网络的输出，同时对目标网络不更新梯度"""
            projection = netFB(x_i)
            feature_t = netP(projection)
            feature_t2 = netP(netFB(x_j))
            sim = loss.get_multi_dis(feature_t, center).to(args.device)
            sim2 = loss.get_multi_dis(feature_t2, center).to(args.device)
            Loss = torch.nn.functional.cross_entropy(sim, labels)
            Loss += torch.nn.functional.cross_entropy(sim2, labels)
            # Loss.requires_grad_(True)
            """update online parameters（更新在线网络的参数）"""
            # feature_t.retain_grad()
            optimizer.zero_grad()  # 清空梯度
            Loss.backward()  # 反向传播
            optimizer.step()  # 优化在线网络参数

            if step % 5 == 0:  # 打印训练中的情况
                print(f"Epoch [{epoch}/{args.proepoch}]; Step [{step}/{len(source_loader)}]:\tLoss: {Loss.item()}")

            """
            update target parameters（更新目标网络的参数）
            target_parameter <=== target_parameter * beta + (1 - beta) * online_parameter
            """
            with torch.no_grad():
                feature = netP(netFB(x_i))
            for i in range(len(labels)):
                if labels[i] == 0:
                    center0.append(feature[i].tolist())
                elif labels[i] == 1:
                    center1.append(feature[i].tolist())
                elif labels[i] == 2:
                    center2.append(feature[i].tolist())
                # elif labels[i] == 3:
                #     center3.append(feature[i].tolist())
        center0 = torch.tensor(center0).mean(dim=0)
        center1 = torch.tensor(center1).mean(dim=0)
        center2 = torch.tensor(center2).mean(dim=0)
        # center3 = torch.tensor(center3).mean(dim=0)
        center_t = torch.stack([center0, center1, center2], dim=0).to(args.device)
        center = center * args.tau + (1-args.tau) * center_t
    center = center.cpu().detach().numpy()
    sio.savemat(osp.join(args.output_dir,'center.mat'), {'center': center})
        # Create the full target model and save i

        # torch.save(netF.state_dict(), osp.join(args.output_dir, "source_Fp.pt"))
        # torch.save(netB.state_dict(), osp.join(args.output_dir, "source_Bp.pt"))
    torch.save(netP.state_dict(), osp.join(args.output_dir, "source_P.pt"))

def project_w(args):
    dset_loaders = digit_load(args, num=1)
    netF = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=args.class_num).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netFBC = nn.Sequential(netF, netB, netC)
    netFBC.eval()
    for k, v in netFBC.named_parameters():
         v.requires_grad = True
    source_loader2 = dset_loaders["source_w"]

    optimizer2 = torch.optim.Adam([{'params': netFBC.parameters()}], lr=0.001)
    N = 1024
    sim_all = []
    for step, ((x_i, x_j), labels, _) in enumerate(source_loader2):
        if step >= N:
            break
        grad1 = []
        grad2 = []
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)
        labels = labels.to(args.device)
        output = netFBC(x_i)
        loss1 = torch.nn.functional.cross_entropy(output, labels)
        optimizer2.zero_grad()
        loss1.backward()
        # optimizer2.step()
        i = 0
        for name, parms in netFBC.named_parameters():
            # print('-->name:', name)
            # print('-->para:', parms)
            # print('-->grad_requirs:',parms.requires_grad)
            # if i >= 111:
            #     break
            grad = torch.zeros(parms.grad.shape).to(args.device)
            grad = grad.copy_(parms.grad)
            grad1.append(grad)
            i += 1
        # print('i:',i)
        output2 = netFBC(x_j)
        loss2 = torch.nn.functional.cross_entropy(output2, labels)
        optimizer2.zero_grad()
        loss2.backward()
        # optimizer2.step()
        for name, parms in netFBC.named_parameters():
            grad2.append(parms.grad)
            # print('-->grad_value:',parms.grad)
        sim = []
        for i in range(len(grad1)):
            s_i = sim_cos(grad1[i], grad2[i])
            sim.append(s_i)
        sim_all.append(sim)
    sim_all = torch.tensor(sim_all)
    sim_all = sim_all.numpy()
    # sim_all = sim_all[~np.isnan(sim_all).any(axis=1), :]
    print(sim_all.shape)
    sim_all = sim_all.mean(axis=0)
    sim_all = (sim_all - sim_all.min()) / (sim_all.max() - sim_all.min())
    sio.savemat(args.output_dir+'/w.mat', {'w': sim_all})
    print('over')


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def train_target(args):
    dset_loaders = digit_load(args)
    # data = sio.loadmat('center.mat')
    data = sio.loadmat(osp.join(args.output_dir,'center.mat'))
    center = data['center']
    center = torch.from_numpy(center).type(torch.FloatTensor).to(args.device)
    data2 = sio.loadmat(args.output_dir+'/w.mat')
    w = data2['w']
    weight = torch.from_numpy(w).type(torch.FloatTensor).to(args.device)
    weight = weight**2
    print('w_size:', weight.shape)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()
    elif args.dset == 'che':
        netF = network.Res50().cuda()
    elif args.dset == 'shice':
        netF = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=args.class_num).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netP = network.MLP(in_features=256, hidden_features=1024, projection_features=256).cuda()


    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_P.pt'
    netP.load_state_dict(torch.load(args.modelpath))
    netP.eval()
    for k, v in netP.named_parameters():
         v.requires_grad = False
    args.modelpath = args.output_dir + '/source_C.pt'    
    netC.load_state_dict(torch.load(args.modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
         v.requires_grad = False
    netFBC = nn.Sequential(netF, netB, netC)

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    # for k, v in netC.named_parameters():
    #     param_group += [{'params': v, 'lr': args.lr}]
    # for k, v in netP.named_parameters():
    #     param_group += [{'params': v, 'lr': args.lr}]
    # netFB = nn.Sequential(netF, netB)

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"]) * 3/10
    interval_iter = len(dset_loaders["target"])
    # interval_iter = max_iter // args.interval
    iter_num = 0

    cro_entropy_all = []
    Im_loss_all = []
    loss1_all = []
    classifier_loss_all = []
    im_loss_all = []
    loss2_all = []
    net_loss_all = []
    all_loss_all = []
    acc_all = []

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            (inputs_test, inputs2), labels, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, inputs2), labels, tar_idx = iter_test.next()

        labels = labels.cuda()
        inputs_test = inputs_test.cuda()
        inputs2 = inputs2.cuda()

        if inputs_test.size(0) == 1:
            continue

        # if iter_num % interval_iter == 0:
        #     plt.ion()
        #     with torch.no_grad():
        #         netF.eval()
        #         netB.eval()
        #         feature = netB(netF(inputs_test))
        #         label = labels
        #         tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        #         plot_only = args.batch_size
        #         low_dim_embs = tsne.fit_transform(feature.cpu().data.numpy()[:plot_only, :])
        #         label = label.cpu().numpy()[:plot_only]
        #         plot_with_labels(low_dim_embs, label, iter_num)

        netF.eval()
        netB.eval()
        acc1, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
        if iter_num == 0:
            print('test_target:', acc1)
        acc_all.append(acc1)
        netF.train()
        netB.train()


        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            mem_label = obtain_label(dset_loaders['target_te'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()
            netC.train()


        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        t1 = time.time()



        # netFB.eval()
        # for k, v in netFB.named_parameters():
        #      v.requires_grad = False
        # class_loss = nn.CrossEntropyLoss()(outputs_test, labels)
        # optimizer.zero_grad()
        # # classifier_loss.backward()
        # class_loss.backward()
        # optimizer.step()
        # netFB.train()
        # for k, v in netFB.named_parameters():
        #      v.requires_grad = True
        # netC.eval()
        # for k, v in netC.named_parameters():
        #      v.requires_grad = False
        #
        # features_test = netB(netF(inputs_test))
        # outputs_test = netC(features_test)
        features_test = netB(netF(inputs_test))
        features2 = netB(netF(inputs2))
        outputs_test = netC(features_test)

        pro_feature = netP(features_test)
        pro_feature2 = netP(features2)

        sim1 = loss.get_multi_dis(pro_feature, center).to(args.device)
        sim2 = loss.get_multi_dis(pro_feature2, center).to(args.device)
        pred_mlp = mem_label[tar_idx]
        # cro_entropy = 0.5*nn.CrossEntropyLoss()(sim1, pred_mlp)
        cro_entropy = 0.5*args.ent_par*(-sim1 * torch.log(sim2)).mean(0).sum()
        cro_entropy_all.append(cro_entropy.item())


        Im_loss = args.ent_par*torch.mean(loss.Entropy(sim1))
        msoft = sim1.mean(dim=0)
        Im_loss -= torch.sum(-msoft * torch.log(msoft+ 1e-5))
        Im_loss_all.append(Im_loss.item())
        loss1 = cro_entropy + Im_loss
        # loss1 = cro_entropy + 

        loss1_all.append(loss1.item())
        # if iter_num % 2 == 0:
        #     optimizer.zero_grad()
        #     # classifier_loss.backward()
        #     loss1.backward()
        #     optimizer.step()
        #
        # features_test = netB(netF(inputs_test))
        # features2 = netB(netF(inputs2))
        # outputs_test = netC(features_test)
        #
        # pro_feature = netP(features_test)
        # pro_feature2 = netP(features2)
        #
        # sim1 = loss.get_multi_dis(pro_feature, center).to(args.device)
        # sim2 = loss.get_multi_dis(pro_feature2, center).to(args.device)
        # cro_entropy = 0.5*args.ent_par*(-sim1 * torch.log(sim2)).mean(0).sum()
        #
        # Im_loss = torch.mean(loss.Entropy(sim1))
        # msoft = sim1.mean(dim=0)
        # Im_loss -= torch.sum(-msoft * torch.log(msoft+ 1e-5))
        # loss1 = Im_loss + cro_entropy

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = args.cls_par* nn.CrossEntropyLoss()(outputs_test, pred)#*max(1,np.power(0.5,iter_num)*max_iter/(1+iter_num))
            classifier_loss_all.append(classifier_loss.item())
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            im_loss_all.append(im_loss.item())
            loss2 = classifier_loss + im_loss
            loss2_all.append(loss2.item())
            classifier_loss += im_loss
        # net_loss = loss.w_loss(netFBC, iter_num, weight)
        # net_loss_all.append(net_loss.item())

        all_loss = classifier_loss + loss1
        all_loss_all.append(all_loss.item())
        # print('net-loss:',500*net_loss,'all_loss:',all_loss)
        if iter_num % 1 == 0:
            optimizer.zero_grad()
            # classifier_loss.backward()
            all_loss.backward()
            optimizer.step()
        t2 = time.time()
        iter_num += 1
        netF.eval()
        netB.eval()
        inputs_test = inputs_test.cuda()
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        _, predict = torch.max(outputs_test,1)
        labels = labels.cuda()
        acc = torch.sum(torch.squeeze(predict).float() == torch.squeeze(labels)).item() / float(labels.size(0))
        t3 = time.time()
        # print('online_acc:',acc,"train:",t2-t1,"test",t3-t2,"all",t3-t1)
        # netFBC.eval()
        # acc1, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
        # acc_all.append(acc1)
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
            log_str = 'Task——test-target: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()
    with open(args.output_dir+'/lossw16.csv', 'w') as outf:
        writer = csv.writer(outf, dialect='excel')
        writer.writerow(tuple(cro_entropy_all))
        writer.writerow(tuple(Im_loss_all))
        writer.writerow(tuple(loss1_all))
        writer.writerow(tuple(classifier_loss_all))
        writer.writerow(tuple(im_loss_all))
        writer.writerow(tuple(loss2_all))
        writer.writerow(tuple(net_loss_all))
        writer.writerow(tuple(all_loss_all))
        writer.writerow(tuple(acc_all))


    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC

def obtain_label(loader, netF, netB, netC, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs, _ = data[0]
            labels = data[1]
            labels = torch.squeeze(labels)
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy= torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'label-Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    return pred_label.astype('int64')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=60, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='shice', choices=['u2m', 'm2u','s2m','che','shice'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.1)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='test')
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--k-disc', type=int, default=1)
    parser.add_argument('--k-clf', type=int, default=10)
    parser.add_argument('--proepoch', type=int, default=20)
    parser.add_argument('--tau', type=float, default=0.99)
    parser.add_argument('--w_loss', type=float, default=0)
    args = parser.parse_args()
    args.class_num = 3


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.device = torch.device('cuda')
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)
    if not osp.exists(args.output_dir+'/w.mat'):
        project_w(args)
    if not osp.exists(osp.join(args.output_dir + '/source_P.pt')):
        project_target(args)

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)
