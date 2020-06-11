import torch
from torch import nn
import numpy as np


class VGG16(torch.nn.Module):
    def __init__(self, model_path):
        torch.nn.Module.__init__(self)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ReLU = nn.ReLU(inplace=True)

        # load channel index
        f = open('../1_pruning/checkpoint/channel_index.txt')
        lines = f.readlines()
        index_code = []
        channel_number = []
        for line in lines:
            line = line.split(', ')
            line = line[0:-1]  # remove '\n'
            tmp = []
            for item in line:
                tmp.append(int(float(item)))
            index_code.append(tmp)
            channel_number.append(np.sum(tmp))
        f.close()

        # add feature layers
        self.conv1_1 = nn.Conv2d(3, channel_number[0], kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(channel_number[0], channel_number[1], kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(channel_number[1], channel_number[2], kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(channel_number[2], channel_number[3], kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(channel_number[3], channel_number[4], kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(channel_number[4], channel_number[5], kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(channel_number[5], channel_number[6], kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(channel_number[6], channel_number[7], kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(channel_number[7], channel_number[8], kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(channel_number[8], channel_number[9], kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(channel_number[9], channel_number[10], kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(channel_number[10], channel_number[11], kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(channel_number[11], 512, kernel_size=3, stride=1, padding=1)

        # add classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('fc6', nn.Linear(512*7*7, 4096))
        self.classifier.add_module('relu6', nn.ReLU(inplace=True))
        self.classifier.add_module('dropout6', nn.Dropout())

        self.classifier.add_module('fc7', nn.Linear(4096, 4096))
        self.classifier.add_module('relu7', nn.ReLU(inplace=True))
        self.classifier.add_module('dropout7', nn.Dropout())

        self.classifier.add_module('fc8', nn.Linear(4096, 1000))

        model_weight = torch.load(model_path)
        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        count = 0
        i = 0
        ind_old = [0, 1, 2]
        for k, v in model_weight.items():
            if 'AP' in k:
                continue
            if 'conv' in k:
                if 'conv5_3' in k:
                    if 'weight' in k:
                        my_weight[my_keys[i]] = v[:, ind_old, :, :]
                    else:
                        my_weight[my_keys[i]] = v
                else:
                    # conv layer
                    if 'weight' in k:
                        # weight
                        ind_ = np.array(index_code[count]).nonzero()[0]
                        v = v[:, ind_old, :, :]
                        my_weight[my_keys[i]] = v[ind_, :, :, :]
                    else:
                        # bias
                        my_weight[my_keys[i]] = v[ind_]
                        ind_old = ind_
                        count += 1
            else:
                # fc layer
                my_weight[my_keys[i]] = v
            i = i + 1
        self.load_state_dict(my_weight)

    def forward(self, x):
        x = self.ReLU(self.conv1_1(x))
        x = self.maxpool(self.ReLU(self.conv1_2(x)))

        x = self.ReLU(self.conv2_1(x))
        x = self.maxpool(self.ReLU(self.conv2_2(x)))

        x = self.ReLU(self.conv3_1(x))
        x = self.ReLU(self.conv3_2(x))
        x = self.maxpool(self.ReLU(self.conv3_3(x)))

        x = self.ReLU(self.conv4_1(x))
        x = self.ReLU(self.conv4_2(x))
        x = self.maxpool(self.ReLU(self.conv4_3(x)))

        x = self.ReLU(self.conv5_1(x))
        x = self.ReLU(self.conv5_2(x))
        x = self.maxpool(self.ReLU(self.conv5_3(x)))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    VGG16('/home/luojh2/model.pth')
