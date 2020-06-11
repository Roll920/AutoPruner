import torch
from new_model import vgg16_compressed, vgg16_test
import argparse
import torch.backends.cudnn as cudnn
import os
import sys
import time
sys.path.append('../')
from src_code.lmdbdataset import lmdbDataset

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--batch_size', default=500, type=int, help='batch size')
parser.add_argument('--data_base', default='/data/zhangcl/ImageNet', type=str, help='the path of dataset')
parser.add_argument('--gpu_id', default='4,5,6,7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def main(model_path):
    # 1. create compressed model
    vgg16_new = vgg16_compressed(layer_id=args.layer_id, model_path=model_path)
    # Phase 2 : Model setup
    vgg16_new = vgg16_new.cuda()
    vgg16_new = torch.nn.DataParallel(vgg16_new.cuda(), device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    new_model_param = vgg16_new.state_dict()
    torch.save(new_model_param, model_path+'model.pth')
    print('Finished!')
    return vgg16_new


def evaluate():
    # Phase 1: load model
    model = vgg16_test('../checkpoint/model.pth')
    # Phase 2 : Model setup
    model = model.cuda()
    model = torch.nn.DataParallel(model.cuda(), device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

    # Phase 2 : Data Upload
    print('\n[Phase 2] : Data Preperation')
    dset_loaders = {
        'train': torch.utils.data.DataLoader(
            lmdbDataset(os.path.join(args.data_base, 'ILSVRC-train.lmdb'), True),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True),

        'val': torch.utils.data.DataLoader(
            lmdbDataset(os.path.join(args.data_base, 'ILSVRC-val.lmdb'), False),
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True)
    }
    print('data_loader_success!')

    # Phase 3: Validation
    print("\n[Phase 3 : Inference on val")
    criterion = torch.nn.CrossEntropyLoss().cuda()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for batch_idx, (input, target) in enumerate(dset_loaders['val']):  # dset_loaders['val']):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            batch_idx, len(dset_loaders['val']), batch_time=batch_time, loss=losses,
            top1=top1, top5=top5))
        sys.stdout.flush()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


if __name__ == '__main__':
    evaluate()
