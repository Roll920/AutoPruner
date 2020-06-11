import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import gradcheck
import numpy as np


class MyGAP_fc(torch.autograd.Function):
    '''
    Global Average Pooling with batchsize: N*4096 -> 1*4096
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
            grad_input[i, :] = grad_output.data / grad_input.shape[0]
        return Variable(grad_input)


class MyScale_fc(torch.autograd.Function):
    '''
    input: x: 64*4096, scale:4096 ==> x[:, i]*scale[i]
    '''
    @staticmethod
    def forward(self, input_data, scale_vec):
        self.save_for_backward(input_data, scale_vec)
        input_data2 = input_data.clone()
        for i in range(scale_vec.shape[0]):
            input_data2[:, i] = input_data[:, i] * scale_vec[i]
        return input_data2

    @staticmethod
    def backward(self, grad_output):
        input_data, scale_vec = self.saved_tensors
        grad_input = input_data.clone()
        for i in range(scale_vec.shape[0]):
            grad_input[:, i] = grad_output.data[:, i] * scale_vec[i]

        grad_vec = scale_vec.clone()
        for i in range(scale_vec.shape[0]):
            grad_vec[i] = torch.sum(grad_output.data[:, i]*input_data[:, i])

        return Variable(grad_input), Variable(grad_vec)


class MyCS_fc(nn.Module):
    def __init__(self, channels_num):
        super(MyCS_fc, self).__init__()
        self.layer_type = 'MyCS_fc'

        self.fc = nn.Linear(channels_num, channels_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale_factor):
        x_scale = MyGAP_fc.apply(x)  # apply my GAP: N*4096 => 1*4096
        x_scale = self.fc(x_scale)  # 1*4096
        x_scale = torch.squeeze(x_scale)  # 4096
        x_scale = x_scale * scale_factor  # apply scale sigmoid
        x_scale = self.sigmoid(x_scale)
        if not self.training:
            index = (np.sign(x_scale.data.cpu().numpy() - 0.5) + 1) / 2.0
            x_scale.data = torch.FloatTensor(index).cuda()
        x = MyScale_fc.apply(x, x_scale)
        return x, x_scale


if __name__ == '__main__':
    in_ = (Variable(torch.randn(3, 4).double(), requires_grad=True),
           Variable(torch.randn(4).double(), requires_grad=True))
    res = gradcheck(MyScale_fc.apply, in_,  eps=1e-6, atol=1e-4)

    # in_ = (Variable(torch.randn(4, 64).double(), requires_grad=True),)
    # res = gradcheck(MyGAP_fc.apply, in_, eps=1e-6, atol=1e-4)

    print(res)

