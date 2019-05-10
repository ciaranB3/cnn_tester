
import torch
import torch.nn as nn

class LIT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=10.0*torch.autograd.Variable(torch.ones(1,1)), num_bits=4):
        ctx.num_bits = num_bits
        ctx.save_for_backward(x, alpha)
        output = x.clamp(min=0, max=alpha.data[0])
        scale = ((2**num_bits)-1)/alpha
        output = torch.round(output * scale) / scale
        # output = x.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.le(0)] = 0
        grad_input[x.ge(alpha.data[0])] = 0
        # print("Max grad value: {} Min grad value: {}".format(grad_input.max(), grad_input.min()))
        # print("Max input value: {}".format(x.max()))

        grad_inputs_sum = grad_output.clone()
        grad_inputs_sum[x.lt(alpha.data[0])] = 0
        grad_inputs_sum = torch.sum(grad_inputs_sum).expand_as(alpha)
        # print("Sum: {} Cloned Sum: {}".format( torch.sum(grad_input), grad_inputs_sum))
        # print("Backward alpha: {}".format(alpha.data))
        return grad_input, grad_inputs_sum, None

class LITnet(nn.Module):
    def __init__(self, alpha=10.0, bit_res=4):
        super(LITnet, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]), requires_grad=True)
        self.bit_res = bit_res
    def forward(self, x):
        return LIT.apply(x, self.alpha, self.bit_res)

def quantizek(num_bits):
    class quant(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx. input):
        scale = (2.0**num_bits) - 1
        out = torch.round(x * scale) / scale 
        return out 

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

    return quant().apply

# def get_DoReMe_weights(weights, num_bits):
#     if num_bits == 1:
#         return weights

#     q_weights = weights.tanh()
#     q_weights = ( q_weights / (2 * torch.max(q_weights)) ) + 0.5
#     return 2 * quantize(q_weights, num_bits -1)

class doreme_weight_quantize(nn.Module):
    def __init__(self, num_bits):
        super(doreme_weight_quantize, self).__init__()
        # self.num_bits = num_bits
        self.quant = quantizek(num_bits)

    def forward(self, x):
        tanhri = torch.tanh(x)
        tanhri = ( tanhri / ( 2 * torch.max( torch.abs(tanhri) ) ) ) + 0.5
        q_weight = ( 2 * self.quant(tanhri) ) - 1
        return q_weight

# class Conv2D_quant(nn.Module):
#     def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1, dialation=1, groups=1, bias=False, bit_res=4):
#         super(Conv2D_quant, self).__init__()
#         self.myconv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias) 
#         self.bit_res = bit_res
#     def forward(self, x):
#         tmp = self.myconv.weight.clone().data
#         self.myconv.weight.data = get_DoReMe_weights(self.myconv.weight.data, self.bit_res)
#         out = self.myconv(x)
#         self.myconv.weight.data = tmp
#         return out

def conv2d_quant(num_bits):
    class Conv2d_q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, 
                    stride=1, padding=0, dilation=1, groups=1, bias=True):
            super(Conv2d_q, self).__init__(in_channels, out_channels, kernel_size, stride,
                    padding, dilation, groups, bias)
            self.num_bits = num_bits
            self.quant = doreme_weight_quantize(num_bits=num_bits)

        def forward(self, input):
            q_weight = self.quant(self.num_bits)
            return F.conv2d(input, q_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    return Conv2d_q

def linear_quant(num_bits):
    class Linear_q(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_q, self).__init__(in_features, out_features, bias)
            self.num_bits = num_bits
            self.quant = doreme_weight_quantize(num_bits=num_bits)

        def forward(self, input):
            q_weight = self.quant(self.weight)
            return F.linear(input, q_weight, self.bias)

    return Linear_q