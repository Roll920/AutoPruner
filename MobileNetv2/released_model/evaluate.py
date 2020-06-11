import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import os
import sys
import argparse
from torchvision import datasets, transforms
import mobilenetv2
from torchsummaryX import summary

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--num_epochs', default=0, type=int, help='number of training epochs')
parser.add_argument('--lr_decay_epoch', default=10, type=int, help='learning rate decay epoch')
parser.add_argument('--data_base', default='/mnt/ramdisk/ImageNet', type=str, help='the path of dataset')
parser.add_argument('--gpu_id', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--ft_model_path', default='mobilenetv2-pruned.pth',
                    type=str, help='the path of fine tuned model')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
best_prec1 = -1
print(args)


def main():
    global args, best_prec1

    # Phase 1 : Model setup
    print('\n[Phase 2] : Model setup')
    model = mobilenetv2.MobileNetV2(args.ft_model_path)
    model.eval()
    summary(model, torch.zeros((1, 3, 224, 224)))
    model_ft = torch.nn.DataParallel(model.cuda())
    cudnn.benchmark = True
    print("model setup success!")

    # Phase 2 : Data Load
    # Data pre-processing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.Resize(256),
            # transforms.RandomCrop((224, 224)),
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
    val_loader = torch.utils.data.DataLoader(dsets['val'], batch_size=args.batch_size, shuffle=False, num_workers=8,
                                             pin_memory=True)
    print('data_loader_success!')

    # Phase 3: fine_tune model
    print('\n[Phase 3] : Model fine tune')
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    validate(val_loader, model_ft, criterion)


def validate(val_loader, model, criterion):
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
        output = model(input)
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
