"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
import torch
from . import my_op

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, block_id, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup

        self.ReLU = nn.ReLU6(inplace=True)

        if expand_ratio == 1:
            self.conv1 = nn.Conv2d(inp, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn2 = nn.BatchNorm2d(oup)
        else:
            self.block_id = block_id
            self.activation_size_list = [112, 56, 56, 28, 28, 28, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7]
            self.AP = my_op.APLayer(hidden_dim, hidden_dim, activation_size=self.activation_size_list[block_id], max_ks=2,
                                    layer_id=block_id)
            # hidden layer of each block
            # 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960]
            self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)
            self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)

        self.index_code = []  # the generated index code
        self.scale_factor = 0.1
        self.channel_index = []  # binary index code for evaluation

    def forward(self, x):
        output = x
        if self.expand_ratio == 1:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
        else:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x_scale = self.AP(x, self.scale_factor, self.channel_index)
            self.index_code = x_scale
            x = my_op.MyScale.apply(x, x_scale)
            x = self.ReLU(self.bn2(self.conv2(x)))
            x = my_op.MyScale.apply(x, x_scale)
            x = self.bn3(self.conv3(x))

        if self.identity:
            return x + output
        else:
            return x


class MobileNetV2(nn.Module):
    def __init__(self, model_path, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        block_id = -1
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(block_id, input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
                block_id += 1
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights(model_path)

    def forward(self, x, scale_factor=1.0, channel_index=None):
        self.set_scale_factor(scale_factor)
        if not self.training:
            self.set_channel_index(channel_index)
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        index_code = self.get_index_code()
        return x, index_code

    def set_scale_factor(self, scale_factor):
        for item in self.features._modules:
            if item == '0' or item == '1':
                continue  # pass the first two blocks
            block = self.features._modules[item]
            block.scale_factor = scale_factor

    def set_channel_index(self, channel_index):
        for item in self.features._modules:
            if item == '0' or item == '1':
                continue  # pass the first two blocks
            block = self.features._modules[item]
            block.channel_index = channel_index

    def get_index_code(self):
        index_code = []
        for item in self.features._modules:
            if item == '0' or item == '1':
                continue  # pass the first two blocks
            block = self.features._modules[item]
            index_code.append(block.index_code)
        return index_code

    def _initialize_weights(self, model_path):
        model_weight = torch.load(model_path)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        my_weight = self.state_dict()
        my_keys = list(my_weight)
        new_keys = []
        for item in my_keys:
            if 'AP' not in item:
                new_keys.append(item)
        for i, (k, v) in enumerate(model_weight.items()):
            my_weight[new_keys[i]] = v
        self.load_state_dict(my_weight)


def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_path = '/mnt/data3/luojh/project/6_CURL/Journal/pretrained_model/ImageNet/mobilenetv2_1.0-0c6065bc.pth'
    model = MobileNetV2(model_path).cuda()
    input = torch.zeros((1, 3, 224, 224)).cuda()
    output = model(input)
    a=1
