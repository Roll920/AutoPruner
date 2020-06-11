import torch
from . import my_op
from torch import nn


class NetworkNew(torch.nn.Module):
    def __init__(self, layer_id=0):
        torch.nn.Module.__init__(self)
        model_weight = torch.load('checkpoint/model.pth')
        channel_length = list()
        channel_length.append(3)
        for k, v in model_weight.items():
            if 'bias' in k:
                channel_length.append(v.size()[0])

        self.feature_1 = nn.Sequential()
        self.feature_2 = nn.Sequential()
        self.classifier = nn.Sequential()
        # add channel selection layers
        ks_dict = {0: 224, 1: 224, 2: 112, 3: 112, 4: 56, 5: 56, 6: 56, 7: 28, 8: 28, 9: 28, 10: 14, 11: 14, 12: 14}
        self.CS = my_op.MyCS(channel_length[layer_id+1], activation_size=ks_dict[layer_id], max_ks=2)

        conv_names = {0: 'conv1_1', 1: 'conv1_2', 2: 'conv2_1', 3: 'conv2_2', 4: 'conv3_1', 5: 'conv3_2', 6: 'conv3_3',
                      7: 'conv4_1', 8: 'conv4_2', 9: 'conv4_3', 10: 'conv5_1', 11: 'conv5_2', 12: 'conv5_3'}
        relu_names = {0: 'relu1_1', 1: 'relu1_2', 2: 'relu2_1', 3: 'relu2_2', 4: 'relu3_1', 5: 'relu3_2', 6: 'relu3_3',
                      7: 'relu4_1', 8: 'relu4_2', 9: 'relu4_3', 10: 'relu5_1', 11: 'relu5_2', 12: 'relu5_3'}
        pool_names = {1: 'pool1', 3: 'pool2', 6: 'pool3', 9: 'pool4', 12: 'pool5'}
        pooling_layer_id = [1, 3, 6, 9, 12]

        # add feature_1 and feature_2 layers
        for i in range(13):
            if i < layer_id:
                self.feature_1.add_module(conv_names[i],
                                          nn.Conv2d(channel_length[i], channel_length[i + 1], kernel_size=3, stride=1,
                                                    padding=1))
                self.feature_1.add_module(relu_names[i], nn.ReLU(inplace=True))
                if i in pooling_layer_id:
                    self.feature_1.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))
            elif i == layer_id:
                self.feature_1.add_module(conv_names[i],
                                          nn.Conv2d(channel_length[i], channel_length[i + 1], kernel_size=3, stride=1,
                                                    padding=1))
                self.feature_1.add_module(relu_names[i], nn.ReLU(inplace=True))
                if i in pooling_layer_id:
                    self.feature_2.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))
            elif i > layer_id:
                self.feature_2.add_module(conv_names[i],
                                          nn.Conv2d(channel_length[i], channel_length[i + 1], kernel_size=3, stride=1,
                                                    padding=1))
                self.feature_2.add_module(relu_names[i], nn.ReLU(inplace=True))
                if i in pooling_layer_id:
                    self.feature_2.add_module(pool_names[i], nn.MaxPool2d(kernel_size=2, stride=2))

        # add classifier
        self.classifier.add_module('fc6', nn.Linear(channel_length[13] * 7 * 7, channel_length[14]))
        self.classifier.add_module('relu6', nn.ReLU(inplace=True))
        self.classifier.add_module('dropout6', nn.Dropout())

        self.classifier.add_module('fc7', nn.Linear(channel_length[14], channel_length[15]))
        self.classifier.add_module('relu7', nn.ReLU(inplace=True))
        self.classifier.add_module('dropout7', nn.Dropout())

        self.classifier.add_module('fc8', nn.Linear(channel_length[15], channel_length[16]))

        # load pretrain model weights
        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        for k, v in model_weight.items():
            name = k.split('.')
            name = 'feature_1.'+name[2]+'.'+name[3]
            if name in my_keys:
                my_weight[name] = v

            name = k.split('.')
            name = 'feature_2.' + name[2] + '.' + name[3]
            if name in my_keys:
                my_weight[name] = v

            name = k[7:]
            if name in my_keys:
                my_weight[name] = v
        self.load_state_dict(my_weight)

    def forward(self, x, scale_factor=1.0, channel_index=None):
        x = self.feature_1(x)
        x, scale_vector = self.CS(x, scale_factor, channel_index)
        x = self.feature_2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, scale_vector


class Vgg16(torch.nn.Module):
    def __init__(self, model_path):
        torch.nn.Module.__init__(self)
        self.feature_1 = nn.Sequential()
        self.classifier = nn.Sequential()

        # add feature layers
        self.feature_1.add_module('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu1_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv2_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu2_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv2_2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu2_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv3_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu3_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv3_2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu3_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv3_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu3_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool3', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv4_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu4_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv4_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu4_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv4_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu4_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_1.add_module('conv5_1', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu5_1', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv5_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu5_2', nn.ReLU(inplace=True))
        self.feature_1.add_module('conv5_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.feature_1.add_module('relu5_3', nn.ReLU(inplace=True))
        self.feature_1.add_module('pool5', nn.MaxPool2d(kernel_size=2, stride=2))

        # add classifier
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
        for k, v in model_weight.items():
            my_weight[my_keys[count]] = v
            count += 1
        self.load_state_dict(my_weight)

    def forward(self, x):
        x = self.feature_1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
