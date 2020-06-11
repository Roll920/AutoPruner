import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import gradcheck
import math


class MyGAP(torch.autograd.Function):
    '''
    Global Average Pooling with batchsize: N*512*14*14 -> 1*512*14*14
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.mean(dim=0, keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        batch_size, num_channels, h, w = input.size()
        grad_input = grad_output.div(batch_size).expand(batch_size, num_channels, h, w)
        return grad_input


class MyScale(torch.autograd.Function):
    '''
    input: x: 64*512*7*7, scale:512 ==> x[:, i, :, :]*scale[i]
    '''
    @staticmethod
    def forward(self, input_data, scale_vec):
        self.save_for_backward(input_data, scale_vec)
        batch_size, num_channels, h, w = input_data.size()
        scale_vec = scale_vec.view([1, num_channels, 1, 1]).expand(input_data.size())
        return input_data.mul(scale_vec)

    @staticmethod
    def backward(self, grad_output):
        input_data, scale_vec = self.saved_tensors
        batch_size, num_channels, h, w = input_data.size()
        scale_vec = scale_vec.view([1, num_channels, 1, 1]).expand(input_data.size())
        grad_input = grad_output.mul(scale_vec)

        grad_vec = grad_output.mul(input_data).sum(-1).sum(-1).sum(0)

        return grad_input, grad_vec


# AutoPruner layer
class APLayer(nn.Module):
    def __init__(self, in_num, out_num, activation_size=14, max_ks=2, layer_id=0):
        super(APLayer, self).__init__()
        self.layer_type = 'APLayer'
        self.id = layer_id
        self.conv = nn.Conv2d(in_num, out_num,
                              kernel_size=int(activation_size / max_ks), stride=1, padding=0)
        self.map = nn.MaxPool2d(kernel_size=max_ks, stride=max_ks)
        self.sigmoid = nn.Sigmoid()
        # n = int(activation_size / max_ks) * int(activation_size / max_ks) * channels_num
        # self.conv.weight.data.normal_(0, 10 * math.sqrt(2.0 / n))
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, scale_factor, channel_index=None):

        x_scale = MyGAP.apply(x)  # apply my GAP: N*512*14*14 => 1*512*14*14
        x_scale = self.map(x_scale)  # apply MAP: 1*512*14*14 => 1*512*7*7
        x_scale = self.conv(x_scale)  # 1*512*1*1

        x_scale = torch.squeeze(x_scale)  # 512
        x_scale = x_scale * scale_factor  # apply scale sigmoid
        x_scale = self.sigmoid(x_scale)

        if not self.training:
            x_scale.data = torch.FloatTensor(channel_index[self.id]).cuda()
        return x_scale


if __name__ == '__main__':
    in_ = (Variable(torch.randn(1, 1, 3, 3).double(), requires_grad=True),
           Variable(torch.randn(1).double(), requires_grad=True))
    res = gradcheck(MyScale.apply, in_,  eps=1e-6, atol=1e-4)

    # in_ = (Variable(torch.randn(2, 64, 3, 3).double(), requires_grad=True),)
    # res = gradcheck(MyGAP.apply, in_, eps=1e-6, atol=1e-4)
    print(res)

