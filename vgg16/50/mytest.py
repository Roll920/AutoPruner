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

from src_code import Network_FT
from src_code.lmdbdataset import lmdbDataset

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', default=2, type=int, help='number of training epochs')
parser.add_argument('--lr_decay_epoch', default=10, type=int, help='learning rate decay epoch')
parser.add_argument('--data_base', default='/data/zhangcl/ImageNet', type=str, help='the path of dataset')
parser.add_argument('--ft_model_path', default='/home/luojh2/.torch/models/vgg16-397923af.pth',
                    type=str, help='the path of fine tuned model')
parser.add_argument('--gpu_id', default='4,5,6,7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--layer_id', default=7, type=int, help='the id of compressed layer, starting from 0')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--compression_rate', default=0.2, type=float, help='the percentage of 1 in compressed model')
parser.add_argument('--channel_index_range', default=20, type=int, help='the range to calculate channel index')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
best_prec1 = -1
print(args)


def main():
    global args, best_prec1
    # Phase 1 : Data Upload
    print('\n[Phase 1] : Data Preperation')
    train_loader = torch.utils.data.DataLoader(
            lmdbDataset(os.path.join(args.data_base, 'ILSVRC-train.lmdb'), True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            lmdbDataset(os.path.join(args.data_base, 'ILSVRC-val.lmdb'), False),
            batch_size=args.batch_size,
            num_workers=16,
            pin_memory=True)
    print('data_loader_success!')

    # Phase 2 : Model setup
    print('\n[Phase 2] : Model setup')
    if args.layer_id == 0:
        model_ft = Network_FT.Vgg16(args.ft_model_path).cuda()
        model_ft = torch.nn.DataParallel(model_ft)
        model_param = model_ft.state_dict()
        torch.save(model_param, 'checkpoint/model.pth')

    model_ft = Network_FT.NetworkNew(args.layer_id).cuda()
    weight = torch.load('checkpoint/layer_7/model.pth')
    model_ft = torch.nn.DataParallel(model_ft)
    model_ft.load_state_dict(weight)
    cudnn.benchmark = True
    print("model setup success!")

    # Phase 3: fine_tune model
    print('\n[Phase 3] : Model fine tune')
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    scale_factor = 9.0

    for epoch in range(args.start_epoch, args.num_epochs):
        adjust_learning_rate(optimizer, epoch, 1)

        # train for one epoch
        channel_index, scale_factor = train(train_loader, model_ft, criterion, optimizer, epoch, scale_factor)

        # evaluate on validation set
        prec1 = validate(val_loader, model_ft, criterion, channel_index)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            folder_path = 'checkpoint/layer_' + str(args.layer_id)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            torch.save(model_ft.state_dict(), folder_path+'/model.pth')
            torch.save(channel_index, folder_path+'/channel_index.pth')


def train(train_loader, model, criterion, optimizer, epoch, scale_factor):
    gpu_num = torch.cuda.device_count()
    scale_factor_mul = math.pow(100, 1.0/(args.num_epochs*len(train_loader)))
    reg_lambda = 100

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    channel_index_list = list()
    channel_index = 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        scale_factor = scale_factor * scale_factor_mul
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        # compute output
        output, scale_vec = model(input_var, scale_factor)
        loss = criterion(output, target_var)
        loss = loss + float(reg_lambda) * (
                scale_vec.norm(1) / float(scale_vec.size(0)) - args.compression_rate) ** 2

        # compute channel index
        tmp = scale_vec.data.cpu().numpy().reshape(gpu_num, -1).mean(0)
        channel_index_list.append(tmp.copy())
        if len(channel_index_list) == args.channel_index_range:
            channel_index_list = np.array(channel_index_list)
            tmp = channel_index_list[0, :]
            print('first 5 values: [{0:.6f}, {1:.6f}, {2:.6f}, {3:.6f}, {4:.6f}]'.format(tmp[0], tmp[1], tmp[2], tmp[3],
                                                                                         tmp[4]))
            tmp2 = channel_index_list.sum(axis=0)
            tmp2 = tmp2 / args.channel_index_range
            for tmp_i in range(len(channel_index_list)):
                channel_index_list[tmp_i] = (np.sign(channel_index_list[tmp_i] - 0.5) + 1) / 2.0
            tmp = channel_index_list.sum(axis=0)
            tmp = tmp / args.channel_index_range
            channel_index = (np.sign(tmp - 0.5) + 1) / 2.0  # to 0-1 binary
            real_pruning_rate = 100.0 * np.sum(tmp2 < 10**-6) / len(tmp2)
            binary_pruning_rate = 100.0 * np.sum(channel_index < 10**-6) / len(channel_index)
            tmp[tmp == 0] = 1
            channel_inconsistency = 100.0 * np.sum(tmp != 1) / len(tmp)
            print("pruning rate (real/binary): {0:.4f}%/{1:.4f}%, index inconsistency: {2:.4f}%".format(
                real_pruning_rate, binary_pruning_rate, channel_inconsistency))
            channel_index_list = list()
            reg_lambda = 100.0 * np.abs(binary_pruning_rate/100.0 - 1 + args.compression_rate)
            sys.stdout.flush()

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
                  'scale_factor: {3:.4f}\t'
                  'reg_lambda: {4:.4f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), scale_factor, reg_lambda, batch_time=batch_time,
                   top1=top1, top5=top5))
            sys.stdout.flush()
    return channel_index, scale_factor


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
        output, _ = model(input_var, 1.0, channel_index)
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
