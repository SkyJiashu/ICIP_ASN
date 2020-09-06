from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os.path as osp
import os
import sys
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as col
import matplotlib.cm as cm
import numpy as np
from torchvision import models

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


EPS = 1e-12

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model_name = 'ResNet50'
        self.base = models.resnet50(pretrained=True)
        self.num_classes = num_classes
        self.num_ftrs = self.base.fc.in_features

        self.base = nn.Sequential(*list(self.base.children())[:-2])

        self.bottleneck = nn.BatchNorm1d(self.num_ftrs)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.num_ftrs, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat_map = self.base(x)  # (b, 2048, 1, 1)

        feat_map_s = (feat_map**2).view(feat_map.shape[0], 2048, -1).sum(2)
        feat_map_base = feat_map.view(feat_map.shape[0], 2048, -1).sum(2)

        global_feat = feat_map_s / (feat_map_base+EPS)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax
        cls_score = self.classifier(feat)

        return cls_score, global_feat, feat

if __name__ == '__main__':

    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    import os

    writer = SummaryWriter()
    # Training settings
    parser = argparse.ArgumentParser(description='CapsNet with MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=3, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--routing_iterations', type=int, default=3)
    # parser.add_argument('--with_reconstruction', action='store_true', default=True)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    R64_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    R32_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    full_dataset = ImageFolder("/home/jiashu/Jiashu/Data/multi_PIE_crop_128/", transform)

    print("full_size :",len(full_dataset))

    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=64, pin_memory = True, num_workers =32,drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(full_dataset, batch_size=64, pin_memory = True, num_workers =32 ,drop_last=True, shuffle=True)

    print("Complete Data Loading")

    model = ResNet50(250)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    loss_fn = nn.CrossEntropyLoss()

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    def accuracy(output, target, topk=(1,)):

        
        maxk = max(topk)

        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        num = 0
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            num = int(correct_k)
            res.append(correct_k.mul_(100.0 / batch_size))
        result = res[0]
        return result, int(num)

    def train(epoch):
        model.train()
        losslist = []
        correctlist = []
        for batch_idx, data_item in enumerate(train_loader):

            data, target = data_item[0], data_item[1]
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            output, global_feat, feat = model(data)
            target = target.long()
            target = target.squeeze() 
            loss = loss_fn(output, target)
            loss.backward()
            losslist.append(loss.data)
            optimizer.step()
            correct, correct_num = accuracy(output, target) 
            correctlist.append(correct)
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data, correct ))
        correct = 0.
        loss = 0.
        i = 0
        for singleresult in correctlist:
            correct = correct + singleresult
            i = i + 1
        for singleloss in losslist:
            loss = loss + singleloss       
        correct = correct/i
        loss = loss/i
      
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/acc', correct, epoch)

    def test():
        torch.cuda.empty_cache()
        model.eval()
        test_loss = 0
        correct = 0
        correctlist = []
        avgcorrect = 0.
        correct_num_total = 0

        with torch.no_grad():
            for data_item in test_loader:

                data, target = data_item[0], data_item[1]
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                target = target.long()
                target = target.squeeze() 
                output, global_feat, feat = model(data)
                test_loss += loss_fn(output, target).data
                correct, correct_num = accuracy(output, target) 
                correct_num_total = correct_num_total + correct_num
                correctlist.append(correct)

        i = 0
        correct = 0
        for singleresult in correctlist:
            correct = correct + singleresult
            i = i + 1
        avgcorrect = correct/i
        avgnum = avgcorrect * len(test_loader.dataset)
        test_loss = test_loss/len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct_num_total, len(test_loader.dataset),
            avgcorrect))

        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', avgcorrect, epoch)

        return test_loss, avgcorrect

    Best_loss = 100000000
    counter_man = 0
    for epoch in range(1, args.epochs + 1):
        
        train(epoch)
        test_loss, avgcorrect= test()
        scheduler.step(test_loss)
        if(avgcorrect > 99.5 and counter_man ==0):
            torch.save(model.state_dict(),
                    '99_F1.pth'.format(str(epoch)))
            counter_man = 1
        if(Best_loss > test_loss):
            Best_loss = test_loss
            torch.save(model.state_dict(),
                    'Bestrouting.pth'.format(str(epoch)))

        torch.save(model.state_dict(),  str(epoch)+'.pth') 

    writer.export_scalars_to_json("./test.json")
    writer.close()
