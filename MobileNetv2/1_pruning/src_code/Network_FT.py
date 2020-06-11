import torch
from . import my_op
from torch import nn


class VGG16(torch.nn.Module):
    def __init__(self, model_path):
        torch.nn.Module.__init__(self)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ReLU = nn.ReLU(inplace=True)

        # add feature layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.AP1_1 = my_op.APLayer(64, activation_size=224, max_ks=2, layer_id=0)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.AP1_2 = my_op.APLayer(64, activation_size=112, max_ks=2, layer_id=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.AP2_1 = my_op.APLayer(128, activation_size=112, max_ks=2, layer_id=2)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.AP2_2 = my_op.APLayer(128, activation_size=56, max_ks=2, layer_id=3)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.AP3_1 = my_op.APLayer(256, activation_size=56, max_ks=2, layer_id=4)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.AP3_2 = my_op.APLayer(256, activation_size=56, max_ks=2, layer_id=5)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.AP3_3 = my_op.APLayer(256, activation_size=28, max_ks=2, layer_id=6)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.AP4_1 = my_op.APLayer(512, activation_size=28, max_ks=2, layer_id=7)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.AP4_2 = my_op.APLayer(512, activation_size=28, max_ks=2, layer_id=8)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.AP4_3 = my_op.APLayer(512, activation_size=14, max_ks=2, layer_id=9)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.AP5_1 = my_op.APLayer(512, activation_size=14, max_ks=2, layer_id=10)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.AP5_2 = my_op.APLayer(512, activation_size=14, max_ks=2, layer_id=11)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

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
        for k, v in model_weight.items():
            if 'AP' in my_keys[count]:
                count = count + 2
            my_weight[my_keys[count]] = v
            count += 1
        self.load_state_dict(my_weight)

    def forward(self, x, scale_factor=1.0, channel_index=None):
        x, s1 = self.AP1_1(self.ReLU(self.conv1_1(x)), scale_factor, channel_index)
        x, s2 = self.AP1_2(self.maxpool(self.ReLU(self.conv1_2(x))), scale_factor, channel_index)

        x, s3 = self.AP2_1(self.ReLU(self.conv2_1(x)), scale_factor, channel_index)
        x, s4 = self.AP2_2(self.maxpool(self.ReLU(self.conv2_2(x))), scale_factor, channel_index)

        x, s5 = self.AP3_1(self.ReLU(self.conv3_1(x)), scale_factor, channel_index)
        x, s6 = self.AP3_2(self.ReLU(self.conv3_2(x)), scale_factor, channel_index)
        x, s7 = self.AP3_3(self.maxpool(self.ReLU(self.conv3_3(x))), scale_factor, channel_index)

        x, s8 = self.AP4_1(self.ReLU(self.conv4_1(x)), scale_factor, channel_index)
        x, s9 = self.AP4_2(self.ReLU(self.conv4_2(x)), scale_factor, channel_index)
        x, s10 = self.AP4_3(self.maxpool(self.ReLU(self.conv4_3(x))), scale_factor, channel_index)

        x, s11 = self.AP5_1(self.ReLU(self.conv5_1(x)), scale_factor, channel_index)
        x, s12 = self.AP5_2(self.ReLU(self.conv5_2(x)), scale_factor, channel_index)
        x = self.maxpool(self.ReLU(self.conv5_3(x)))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]
