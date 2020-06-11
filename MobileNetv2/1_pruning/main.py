import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import sys
import argparse
import numpy as np
import shutil
from torchvision import datasets, transforms
from src_code import mobilenetv2

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', default=5, type=int, help='number of training epochs')
parser.add_argument('--lr_decay_epoch', default=10, type=int, help='learning rate decay epoch')
parser.add_argument('--data_base', default='/mnt/ramdisk/ImageNet', type=str, help='the path of dataset')
parser.add_argument('--ft_model_path', default='/mnt/data3/luojh/project/6_CURL/Journal/pretrained_model/ImageNet/mobilenetv2_1.0-0c6065bc.pth',
                    type=str, help='the path of fine tuned model')
parser.add_argument('--gpu_id', default='4,5,6,7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--compression_rate', default=0.64, type=float, help='the proportion of 1 in compressed model')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--alpha_start', default=0.1, type=float, help='the initial value of alpha in AutoPruner layer')
parser.add_argument('--alpha_end', default=100, type=float, help='the initial value of alpha in AutoPruner layer')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
best_prec1 = -1
alpha = 0
alpha_step = 0

print(args)
# args.ft_model_path = '/home/luojh2/.torch/models/vgg16-397923af.pth'


def main():
    global args, best_prec1, alpha, alpha_step

    # Phase 1 : Model setup
    print('\n[Phase 2] : Model setup')
    model = mobilenetv2.MobileNetV2(args.ft_model_path).cuda()
    print(model)
    model_ft = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    print("model setup success!")

    # Phase 2 : Data Load
    # Data pre-processing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = args.data_base
    print("| Preparing data...")
    dsets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    train_loader = torch.utils.data.DataLoader(dsets['train'], batch_size=args.batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dsets['val'], batch_size=args.batch_size, shuffle=False, num_workers=8,
                                             pin_memory=True)
    print('data_loader_success!')

    # Phase 3: fine_tune model
    print('\n[Phase 3] : Model fine tune')
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()), args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    alpha_step = (args.alpha_end - args.alpha_start)/float(args.num_epochs * len(train_loader))
    alpha = args.alpha_start
    for epoch in range(args.start_epoch, args.num_epochs):
        # adjust_learning_rate(optimizer, epoch, 3)  # reduce lr every 3 epochs

        # train for one epoch
        time1 = time.time()
        channel_index = train(train_loader, model_ft, criterion, optimizer, epoch)
        print('training one epoch takes {0:.3f} seconds.'.format(time.time()-time1))

        # evaluate on validation set
        prec1 = validate(val_loader, model_ft, criterion, channel_index)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
        folder_path = 'checkpoint'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(model_ft.state_dict(), folder_path+'/model.pth')
        fw = open(folder_path+'/channel_index.txt', 'w')
        for item in channel_index:
            for item_ in item:
                fw.write('{0}, '.format(item_))
            fw.write('\n')
        fw.close()


def train(train_loader, model, criterion, optimizer, epoch):
    global alpha_step, alpha
    gpu_num = torch.cuda.device_count()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        alpha = alpha + alpha_step
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.cuda()

        # compute output
        output, scale_vec = model(input, alpha)

        # adjust alpha and reg_lambda
        channel_index = []
        first_value = []
        binary_pruning_rate = []
        index_code = []
        two_side_rate = []
        for item in scale_vec:
            tmp = item.data.cpu().numpy().reshape(gpu_num, -1).mean(0)
            index_code.append(tmp)

        for item in index_code:
            tmp = item
            channel_index.append((np.sign(tmp - 0.5) + 1) / 2.0)
            first_value.append(tmp[0])
            binary_pruning_rate.append(np.sum(tmp < 0.1) / len(tmp))  # The proportion of 0
            two_side_rate.append((np.sum(tmp > 0.9) + np.sum(tmp < 0.1)) / len(tmp))

        if i % args.print_freq == 0:
            print('The first value of each layer: {0}'.format(first_value))
            print('The binary rate of each layer: {0}'.format(binary_pruning_rate))
            print('The two side rate of each layer: {0}'.format(two_side_rate))

        # check the binary rate in the last epoch
        if epoch == args.num_epochs - 1:
            if not all(my_item > 0.9 for my_item in two_side_rate):
                alpha = alpha + 10*alpha_step

        # calculate loss
        loss1 = criterion(output, target)
        loss2 = 0
        for ind_, item in enumerate(scale_vec):
            loss2 += (item.norm(1) / float(item.size(0)) - args.compression_rate) ** 2
        loss = loss1 + 10.0*loss2

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
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
                  'alpha: {3:.4f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, i, len(train_loader), alpha, batch_time=batch_time,
                   top1=top1, top5=top5, loss=losses))
            sys.stdout.flush()

    return channel_index


def validate(val_loader, model, criterion, channel_index):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()

        # compute output
        output, _ = model(input, 1.0, channel_index)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
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
