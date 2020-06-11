import torch
from new_model import vgg16_compressed
import argparse
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
parser.add_argument('--layer_id', default=2, type=int, help='the id of compressed layer, starting from 0')
args = parser.parse_args()
print(args)


def main(model_path):
    # 1. create compressed model
    vgg16_new = vgg16_compressed(layer_id=args.layer_id, model_path=model_path)
    # Phase 2 : Model setup
    vgg16_new = vgg16_new.cuda()
    vgg16_new = torch.nn.DataParallel(vgg16_new.cuda(), device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    new_model_param = vgg16_new.state_dict()
    torch.save(new_model_param, '../checkpoint/model.pth')
    print('Finished!')


if __name__ == '__main__':
    folder_path = '../checkpoint/layer_' + str(args.layer_id)+'/'
    main(folder_path)
