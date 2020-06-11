import torch.nn as nn
import math
import torch
from . import my_op


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, number_list, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(number_list[1], number_list[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(number_list[0])
        self.conv2 = nn.Conv2d(number_list[3], number_list[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(number_list[2])
        self.conv3 = nn.Conv2d(number_list[5], number_list[4], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(number_list[4])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_with_CS(nn.Module):
    expansion = 4

    def __init__(self, number_list, stride=1, downsample=None, ks=1, CS_id=0):
        super(Bottleneck_with_CS, self).__init__()
        self.conv1 = nn.Conv2d(number_list[1], number_list[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(number_list[0])
        self.conv2 = nn.Conv2d(number_list[3], number_list[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(number_list[2])
        self.conv3 = nn.Conv2d(number_list[5], number_list[4], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(number_list[4])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.CS_id = CS_id
        self.channel_index = list()

        if ks == 7:
            mks = 1
        else:
            mks = 2
        self.cs1 = my_op.MyCS(number_list[0], activation_size=ks * stride, max_ks=mks)
        self.cs2 = my_op.MyCS(number_list[2], activation_size=ks, max_ks=mks)
        self.vec1 = None
        self.vec2 = None
        self.scale_factor1 = 1.0
        self.scale_factor2 = 1.0

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.training:
            out, self.vec1 = self.cs1(out, self.scale_factor1)
        else:
            out, self.vec1 = self.cs1(out, self.scale_factor1, self.channel_index[2 * self.CS_id])

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.training:
            out, self.vec2 = self.cs2(out, self.scale_factor2)
        else:
            out, self.vec2 = self.cs2(out, self.scale_factor2, self.channel_index[2 * self.CS_id + 1])

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, group_id, block, layers, num_classes=1000):
        old_weight = torch.load('checkpoint/model.pth')
        channel_number_list = analyse_number(old_weight)

        self.kernel_size = int(56 / (2**group_id))
        self.inplanes = 64
        self.g_id = group_id

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(channel_number_list[0], 0, block, 64, layers[0])
        self.layer2 = self._make_layer(channel_number_list[1], 1, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(channel_number_list[2], 2, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(channel_number_list[3], 3, block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         # m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.weight.data.normal_(0, math.sqrt(1.))
        #         # torch.nn.init.xavier_uniform(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        old_weight = torch.load('checkpoint/model.pth')
        my_weight = self.state_dict()
        my_keys = list(my_weight.keys())
        for k, v in old_weight.items():
            name = ''.join(list(k)[7:])
            if name in my_keys:
                my_weight[name] = v
        self.load_state_dict(my_weight)

    def _make_layer(self, number_list, group_id, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if group_id == self.g_id:
            layers.append(Bottleneck_with_CS(number_list[0], stride, downsample, ks=self.kernel_size, CS_id=0))
        else:
            layers.append(block(number_list[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if group_id == self.g_id:
                if self.g_id == 3 and i == blocks-1:
                    layers.append(block(number_list[i]))
                else:
                    layers.append(Bottleneck_with_CS(number_list[i], ks=self.kernel_size, CS_id=i))
            else:
                layers.append(block(number_list[i]))

        return nn.Sequential(*layers)

    def forward(self, x, channel_index=None):
        if not self.training:
            self.set_channel_index(channel_index)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 128, 64, 56, 56

        x = self.layer1(x)  # 128, 64, 56, 56
        x = self.layer2(x)  # 128, 512, 28, 28
        x = self.layer3(x)  # 128, 1024, 14, 14
        x = self.layer4(x)  # 128, 2048, 7, 7

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        scale_vector = self.get_scale_vector()

        return x, scale_vector

    def set_channel_index(self, channel_index):
        if self.g_id == 0:
            self.layer1[0].channel_index = channel_index
            self.layer1[1].channel_index = channel_index
            self.layer1[2].channel_index = channel_index
        elif self.g_id == 1:
            self.layer2[0].channel_index = channel_index
            self.layer2[1].channel_index = channel_index
            self.layer2[2].channel_index = channel_index
            self.layer2[3].channel_index = channel_index
        elif self.g_id == 2:
            self.layer3[0].channel_index = channel_index
            self.layer3[1].channel_index = channel_index
            self.layer3[2].channel_index = channel_index
            self.layer3[3].channel_index = channel_index
            self.layer3[4].channel_index = channel_index
            self.layer3[5].channel_index = channel_index
        else:
            self.layer4[0].channel_index = channel_index
            self.layer4[1].channel_index = channel_index
            # self.layer4[2].channel_index = channel_index

    def get_scale_vector(self):
        vector_list = list()
        if self.g_id == 0:
            vector_list.append(self.layer1[0].vec1)
            vector_list.append(self.layer1[0].vec2)
            vector_list.append(self.layer1[1].vec1)
            vector_list.append(self.layer1[1].vec2)
            vector_list.append(self.layer1[2].vec1)
            vector_list.append(self.layer1[2].vec2)

        elif self.g_id == 1:
            vector_list.append(self.layer2[0].vec1)
            vector_list.append(self.layer2[0].vec2)
            vector_list.append(self.layer2[1].vec1)
            vector_list.append(self.layer2[1].vec2)
            vector_list.append(self.layer2[2].vec1)
            vector_list.append(self.layer2[2].vec2)
            vector_list.append(self.layer2[3].vec1)
            vector_list.append(self.layer2[3].vec2)
        elif self.g_id == 2:
            vector_list.append(self.layer3[0].vec1)
            vector_list.append(self.layer3[0].vec2)
            vector_list.append(self.layer3[1].vec1)
            vector_list.append(self.layer3[1].vec2)
            vector_list.append(self.layer3[2].vec1)
            vector_list.append(self.layer3[2].vec2)
            vector_list.append(self.layer3[3].vec1)
            vector_list.append(self.layer3[3].vec2)
            vector_list.append(self.layer3[4].vec1)
            vector_list.append(self.layer3[4].vec2)
            vector_list.append(self.layer3[5].vec1)
            vector_list.append(self.layer3[5].vec2)
        else:
            vector_list.append(self.layer4[0].vec1)
            vector_list.append(self.layer4[0].vec2)
            vector_list.append(self.layer4[1].vec1)
            vector_list.append(self.layer4[1].vec2)
            # vector_list.append(self.layer4[2].vec1)
            # vector_list.append(self.layer4[2].vec2)
        return vector_list

    def set_scale_factor(self, sf):
        if self.g_id == 0:
            self.layer1[0].scale_factor1 = sf[0]
            self.layer1[0].scale_factor2 = sf[1]
            self.layer1[1].scale_factor1 = sf[2]
            self.layer1[1].scale_factor2 = sf[3]
            self.layer1[2].scale_factor1 = sf[4]
            self.layer1[2].scale_factor2 = sf[5]
        elif self.g_id == 1:
            self.layer2[0].scale_factor1 = sf[0]
            self.layer2[0].scale_factor2 = sf[1]
            self.layer2[1].scale_factor1 = sf[2]
            self.layer2[1].scale_factor2 = sf[3]
            self.layer2[2].scale_factor1 = sf[4]
            self.layer2[2].scale_factor2 = sf[5]
            self.layer2[3].scale_factor1 = sf[6]
            self.layer2[3].scale_factor2 = sf[7]
        elif self.g_id == 2:
            self.layer3[0].scale_factor1 = sf[0]
            self.layer3[0].scale_factor2 = sf[1]
            self.layer3[1].scale_factor1 = sf[2]
            self.layer3[1].scale_factor2 = sf[3]
            self.layer3[2].scale_factor1 = sf[4]
            self.layer3[2].scale_factor2 = sf[5]
            self.layer3[3].scale_factor1 = sf[6]
            self.layer3[3].scale_factor2 = sf[7]
            self.layer3[4].scale_factor1 = sf[8]
            self.layer3[4].scale_factor2 = sf[9]
            self.layer3[5].scale_factor1 = sf[10]
            self.layer3[5].scale_factor2 = sf[11]
        else:
            self.layer4[0].scale_factor1 = sf[0]
            self.layer4[0].scale_factor2 = sf[1]
            self.layer4[1].scale_factor1 = sf[2]
            self.layer4[1].scale_factor2 = sf[3]
            # self.layer4[2].scale_factor = sf


def analyse_number(weight):
    number_list = list()
    group_list = list()
    layer_list = list()
    old_name = '1.0.'
    old_group = '1'
    for k, v in weight.items():
        if 'layer' in k and'conv' in k:
            current_name = k.split('layer')[1].split('conv')[0]
            current_group = current_name.split('.')[0]
            if current_name != old_name:
                old_name = current_name
                group_list.append(layer_list.copy())
                layer_list = list()
            if current_group != old_group:
                old_group = current_group
                number_list.append(group_list.copy())
                group_list = list()
            layer_list.append(v.size()[0])
            layer_list.append(v.size()[1])
    group_list.append(layer_list.copy())
    number_list.append(group_list.copy())
    return number_list


def NetworkNew(group_id):
    model = ResNet(group_id, Bottleneck, [3, 4, 6, 3])
    return model
