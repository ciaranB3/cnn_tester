from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import my_optimizer

class LIT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=8.0*torch.autograd.Variable(torch.ones(1,1))):
        # alpha, = ctx.saved_tensors
        # print("Alpha: {}".format(alpha.data))
        ctx.save_for_backward(x, alpha)
        return F.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, alpha= ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        print("\n\nNew layer")
        print(x)
        print(grad_input)
        return grad_input

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0
        return grad_input


if __name__ == '__main__':

    dtype = torch.float
    # device = torch.device("cpu")
    device = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 4, 10, 10, 2

    torch.manual_seed(1)

    # Create random Tensors to hold input and outputs.
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Create random Tensors for weights.
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-3
    for t in range(10):
        # To apply our Function, we use Function.apply method. We alias this as 'relu'.
        relu = LIT.apply

        # Forward pass: compute predicted y using operations; we compute
        # ReLU using our custom autograd operation.
        y_pred = relu(x.mm(w1)).mm(w2)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum()
        print(loss.item())

        # Use autograd to compute the backward pass.
        loss.backward()

        # Update weights using gradient descent
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()
