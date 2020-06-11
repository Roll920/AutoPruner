# ************************************************************
# Author : Bumsoo Kim, 2017
# Github : https://github.com/meliketoy/fine-tuning.pytorch
#
# Korea University, Data-Mining Lab
# Deep Convolutional Network Fine tuning Implementation
#
# Description : main.py
# The main code for training classification networks.
# ***********************************************************

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import sys
import argparse
import numpy as np
import shutil
import math
from torchvision import models

from src_code import Network_FT
from src_code.lmdbdataset import lmdbDataset

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', default=8, type=int, help='number of training epochs')
parser.add_argument('--lr_decay_epoch', default=10, type=int, help='learning rate decay epoch')
parser.add_argument('--data_base', default='/data/zhangcl/ImageNet', type=str, help='the path of dataset')
parser.add_argument('--ft_model_path', default='/home/luojh2/.torch/models/resnet50-19c8e357.pth',
                    type=str, help='the path of fine tuned model')
parser.add_argument('--gpu_id', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--group_id', default=0, type=int, help='the id of compressed group, starting from 0')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--compression_rate', default=0.5, type=float, help='the percentage of 1 in compressed model')
parser.add_argument('--channel_index_range', default=20, type=int, help='the range to calculate channel index')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--alpha_range', default=100, type=int, help='the range to calculate channel index')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
best_prec1 = -1
print(args)
resnet_channel_number = [6, 8, 12, 4]
scale_factor_list = None
alpha_index = 0
threshold = 95 * np.ones(resnet_channel_number[args.group_id])


def main():
    global args, best_prec1, scale_factor_list, resnet_channel_number
    # Phase 1 : Data Upload
    print('\n[Phase 1] : Data Preperation')
    train_loader = torch.utils.data.DataLoader(
            lmdbDataset(os.path.join(args.data_base, 'ILSVRC-train.lmdb'), True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            lmdbDataset(os.path.join(args.data_base, 'ILSVRC-val.lmdb'), False),
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True)
    print('data_loader_success!')

    # Phase 2 : Model setup
    print('\n[Phase 2] : Model setup')
    if args.group_id == 0:
        model_ft = models.resnet50(True).cuda()
        model_ft = torch.nn.DataParallel(model_ft)
        model_param = model_ft.state_dict()
        torch.save(model_param, 'checkpoint/model.pth')

    model_ft = Network_FT.NetworkNew(args.group_id).cuda()
    model_ft = torch.nn.DataParallel(model_ft)
    cudnn.benchmark = True
    print("model setup success!")

    # Phase 3: fine_tune model
    print('\n[Phase 3] : Model fine tune')
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    tmp = np.linspace(1, 100, int(args.num_epochs * len(train_loader) / args.alpha_range))
    scale_factor_list = np.ones([resnet_channel_number[args.group_id], len(tmp)])
    for tmp_i in range(resnet_channel_number[args.group_id]):
        scale_factor_list[tmp_i, :] = tmp.copy()
    reg_lambda = 10.0 * np.ones(resnet_channel_number[args.group_id])
    for epoch in range(args.start_epoch, args.num_epochs):
        adjust_learning_rate(optimizer, epoch, int(args.num_epochs/2.0))
        # train for one epoch
        channel_index, reg_lambda = train(train_loader, model_ft, criterion, optimizer, epoch, reg_lambda)

        # evaluate on validation set
        prec1 = validate(val_loader, model_ft, criterion, channel_index)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            folder_path = 'checkpoint/group_' + str(args.group_id)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(model_ft.state_dict(), folder_path+'/model.pth')
            if args.group_id == 3:
                tmp = channel_index[0].copy()
                tmp[:] = 1.0
                channel_index.append(tmp.copy())
                channel_index.append(tmp.copy())
            torch.save(channel_index, folder_path+'/channel_index.pth')


def train(train_loader, model, criterion, optimizer, epoch, reg_lambda):
    global resnet_channel_number, scale_factor_list, alpha_index, threshold
    gpu_num = torch.cuda.device_count()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    channel_index_list = list()
    channel_index_binary = list()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if i % args.alpha_range == 0:
            if alpha_index == scale_factor_list.shape[1]:
                alpha_index = alpha_index - 1
            scale_factor = scale_factor_list[:, alpha_index]
            alpha_index = alpha_index + 1

        model.module.set_scale_factor(scale_factor)
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        # compute output
        output, scale_vec = model(input_var)
        loss = criterion(output, target_var)
        for vec_i in range(len(scale_vec)):
            loss = loss + float(reg_lambda[vec_i]) * (
                    scale_vec[vec_i].norm(1) / float(scale_vec[vec_i].size(0)) - args.compression_rate) ** 2

        # compute channel index
        channel_index_sublist = list()
        for vec_i in range(len(scale_vec)):
            tmp = scale_vec[vec_i].data.cpu().numpy().reshape(gpu_num, -1).mean(0)
            channel_index_sublist.append(tmp.copy())
            if i == 0:
                print('first 5 values in layer {0}: [{1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}, {5:.6f}]'.format(int(vec_i),
                                                                                                          tmp[0],
                                                                                                          tmp[1],
                                                                                                          tmp[2],
                                                                                                          tmp[3],
                                                                                                          tmp[4]))
        channel_index_list.append(channel_index_sublist.copy())

        if len(channel_index_list) == args.channel_index_range:
            channel_index_binary = list()
            for vec_i in range(len(scale_vec)):
                tmp = list()
                for tmp_i in range(len(channel_index_list)):
                    tmp_a = channel_index_list[tmp_i][vec_i]
                    tmp_a = (np.sign(tmp_a - 0.5) + 1) / 2.0  # to 0-1 binary
                    tmp.append(tmp_a)
                tmp = np.array(tmp).sum(axis=0)
                tmp = tmp / args.channel_index_range
                tmp_value = channel_index_list[0][vec_i]
                print(
                    'first 5 values in layer {0}: [{1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}, {5:.6f}]'.format(int(vec_i),
                                                                                                        tmp_value[0],
                                                                                                        tmp_value[1],
                                                                                                        tmp_value[2],
                                                                                                        tmp_value[3],
                                                                                                        tmp_value[4]))
                channel_index = (np.sign(tmp - 0.5) + 1) / 2.0  # to 0-1 binary
                channel_index_binary.append(channel_index.copy())
                binary_pruning_rate = 100.0 * np.sum(channel_index == 0) / len(channel_index)

                if binary_pruning_rate >= threshold[vec_i]:
                    scale_factor_list[vec_i, :] = scale_factor_list[vec_i, :] + 1
                    threshold[vec_i] = threshold[vec_i] - 5
                    if threshold[vec_i] < 100 - 100 * args.compression_rate:
                        threshold[vec_i] = 100 - 100 * args.compression_rate
                    print('threshold in layer %d is %d' % (int(vec_i), int(threshold[vec_i])))

                two_side_rate = (np.sum(tmp_value > 0.8) + np.sum(tmp_value < 0.2)) / len(tmp_value)
                if two_side_rate < 0.9 and alpha_index >= int(scale_factor_list.shape[1] / args.num_epochs):
                    scale_factor_list[vec_i, :] = scale_factor_list[vec_i, :] + 1

                reg_lambda[vec_i] = 100.0 * np.abs(binary_pruning_rate/100.0 - 1 + args.compression_rate)
                tmp[tmp == 0] = 1
                channel_inconsistency = 100.0 * np.sum(tmp != 1) / len(tmp)
                print(
                    "[{0}] pruning rate: {1:.4f}%, inconsistency: {2:.4f}%, reg_lambda: {3:.4f}, scale_factor: {4:.4f}, two_side: {5:.4f}".format(
                        int(vec_i), binary_pruning_rate, channel_inconsistency, reg_lambda[vec_i], scale_factor[vec_i], two_side_rate))
                sys.stdout.flush()
            channel_index_list = list()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch[{0}]: [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   top1=top1, top5=top5))
            print('+--------------------------------------------------------------------------------------------------+')
            sys.stdout.flush()

    return channel_index_binary, reg_lambda


def validate(val_loader, model, criterion, channel_index):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, _ = model(input_var, channel_index)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, epoch_num):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // epoch_num))
    print('| Learning Rate = %f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
