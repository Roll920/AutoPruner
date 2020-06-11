import torch.nn as nn
import torch
import numpy as np


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


class ResNet(nn.Module):

    def __init__(self, group_id, block, layers, num_classes=1000):
        folder_path = '../checkpoint/group_' + str(group_id)
        old_weight = torch.load(folder_path+'/model.pth')
        channel_index = torch.load(folder_path+'/channel_index.pth')
        channel_number_list = analyse_number(old_weight)

        for i in range(int(len(channel_index)/2)):
            new_num = np.where(channel_index[2 * i] != 0)[0]
            new_num_1 = int(new_num.shape[0])
            new_num = np.where(channel_index[2 * i + 1] != 0)[0]
            new_num_2 = int(new_num.shape[0])
            channel_number_list[group_id][i][0] = new_num_1
            channel_number_list[group_id][i][2] = new_num_2
            channel_number_list[group_id][i][3] = new_num_1
            channel_number_list[group_id][i][5] = new_num_2

        self.inplanes = 64

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(channel_number_list[0], block, 64, layers[0])
        self.layer2 = self._make_layer(channel_number_list[1], block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(channel_number_list[2], block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(channel_number_list[3], block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        my_weight = self.state_dict()
        ci_count = 0
        ci_1 = 0
        ci_2 = 0
        for k, v in my_weight.items():
            name = 'module.' + k
            if 'layer'+str(group_id+1) in name and 'downsample' not in name:
                name_tmp = name.split('.')
                if '1' in name_tmp[3]:
                    if 'conv' in name:
                        ci_1 = torch.cuda.LongTensor(np.where(channel_index[ci_count] != 0)[0])
                        ci_count += 1
                        my_weight[k] = old_weight[name][ci_1, :, :, :]
                    else:
                        my_weight[k] = old_weight[name][ci_1]
                elif '2' in name_tmp[3]:
                    if 'conv' in name:
                        ci_2 = torch.cuda.LongTensor(np.where(channel_index[ci_count] != 0)[0])
                        ci_count += 1
                        my_weight[k] = old_weight[name][ci_2, :, :, :]
                        my_weight[k] = my_weight[k][:, ci_1, :, :]
                    else:
                        my_weight[k] = old_weight[name][ci_2]
                elif '3' in name_tmp[3]:
                    if 'conv' in name:
                        my_weight[k] = old_weight[name][:, ci_2, :, :]
                    else:
                        my_weight[k] = old_weight[name]
                else:
                    print('error!')
            else:
                my_weight[k] = old_weight[name]
        self.load_state_dict(my_weight)

    def _make_layer(self, number_list, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(number_list[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(number_list[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def analyse_number(weight):
    number_list = list()
    group_list = list()
    layer_list = list()
    old_name = '1.0.'
    old_group = '1'
    for k, v in weight.items():
        if 'layer' in k and'conv' in k and 'cs' not in k:
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


class ResNet_test(nn.Module):

    def __init__(self, model_path, block, layers, num_classes=1000):
        old_weight = torch.load(model_path)
        channel_number_list = analyse_number(old_weight)

        self.inplanes = 64

        super(ResNet_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(channel_number_list[0], block, 64, layers[0])
        self.layer2 = self._make_layer(channel_number_list[1], block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(channel_number_list[2], block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(channel_number_list[3], block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        my_weight = self.state_dict()
        for k, v in my_weight.items():
            name = 'module.' + k
            my_weight[k] = old_weight[name]
        self.load_state_dict(my_weight)

    def _make_layer(self, number_list, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(number_list[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(number_list[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def NetworkNew_test(model_path):
    model = ResNet_test(model_path, Bottleneck, [3, 4, 6, 3])
    return model
