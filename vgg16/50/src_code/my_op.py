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
        input = torch.mean(input, dim=0, keepdim=True)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = input[0].clone()
        for i in range(grad_input.shape[0]):
            grad_input[i, :, :, :] = grad_output.data / grad_input.shape[0]
        return Variable(grad_input)


class MyScale(torch.autograd.Function):
    '''
    input: x: 64*512*7*7, scale:512 ==> x[:, i, :, :]*scale[i]
    '''
    @staticmethod
    def forward(self, input_data, scale_vec):
        self.save_for_backward(input_data, scale_vec)
        input_data2 = input_data.clone()
        for i in range(scale_vec.shape[0]):
            input_data2[:, i, :, :] = input_data[:, i, :, :] * scale_vec[i]
        return input_data2

    @staticmethod
    def backward(self, grad_output):
        input_data, scale_vec = self.saved_tensors
        grad_input = input_data.clone()
        for i in range(scale_vec.shape[0]):
            grad_input[:, i, :, :] = grad_output.data[:, i, :, :] * scale_vec[i]

        grad_vec = scale_vec.clone()
        for i in range(scale_vec.shape[0]):
            grad_vec[i] = torch.sum(grad_output.data[:, i, :, :]*input_data[:, i, :, :])

        return Variable(grad_input), Variable(grad_vec)


class MyCS(nn.Module):
    def __init__(self, channels_num, activation_size=14, max_ks=2):
        super(MyCS, self).__init__()
        self.layer_type = 'MyCS'

        self.conv = nn.Conv2d(channels_num, channels_num,
                              kernel_size=int(activation_size / max_ks), stride=1, padding=0)
        self.map = nn.MaxPool2d(kernel_size=max_ks, stride=max_ks)
        self.sigmoid = nn.Sigmoid()
        # self.conv.weight.data.normal_(0, 0.005)
        n = int(activation_size / max_ks) * int(activation_size / max_ks) * channels_num
        self.conv.weight.data.normal_(0, 10*math.sqrt(2.0/n))

        # torch.nn.init.xavier_normal(self.conv.weight)
        # torch.nn.init.constant(self.conv.bias, 0)

    def forward(self, x, scale_factor, channel_index=None):

        x_scale = MyGAP.apply(x)  # apply my GAP: N*512*14*14 => 1*512*14*14
        x_scale = self.map(x_scale)  # apply MAP: 1*512*14*14 => 1*512*7*7
        x_scale = self.conv(x_scale)  # 1*512*1*1

        x_scale = torch.squeeze(x_scale)  # 512
        x_scale = x_scale * scale_factor  # apply scale sigmoid
        x_scale = self.sigmoid(x_scale)

        if not self.training:
            x_scale.data = torch.FloatTensor(channel_index).cuda()

        x = MyScale.apply(x, x_scale)
        return x, x_scale


if __name__ == '__main__':
    # in_ = (Variable(torch.randn(1, 1, 3, 3).double(), requires_grad=True),
    #        Variable(torch.randn(1).double(), requires_grad=True))
    # res = gradcheck(MyScale.apply, in_,  eps=1e-6, atol=1e-4)

    in_ = (Variable(torch.randn(2, 64, 3, 3).double(), requires_grad=True),)
    res = gradcheck(MyGAP.apply, in_, eps=1e-6, atol=1e-4)
    print(res)

