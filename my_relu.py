from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
import my_optimizer
import random

class LIT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=10.0*torch.autograd.Variable(torch.ones(1,1))):
        # x, alpha = ctx.saved_tensors
        print("Forward alpha: {}".format(alpha.data))
        print("Max input value: {}".format(x.max()))
        ctx.save_for_backward(x, alpha)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, alpha = ctx.saved_tensors
        # alpha =  alpha-alpha*1000
        grad_input = grad_output.clone()
        grad_input[x <= 0] = 0
        grad_input[x >= alpha] = 0
        print("Max grad value: {} Min grad value: {}".format(grad_input.max(), grad_input.min()))
        print("Max input value: {}".format(x.max()))

        grad_inputs_sum = grad_output.clone()
        grad_inputs_sum[x<alpha] = 0
        print("Values: {}".format(grad_inputs_sum))
        grad_inputs_sum = torch.sum(grad_inputs_sum)*torch.ones(1).to(device)
        print("Sum: {} Cloned Sum: {}".format( torch.sum(grad_input), grad_inputs_sum))
        # print("Backward alpha: {}".format(alpha.data))
        return grad_input, grad_inputs_sum

class LITnet(nn.Module):
    def __init__(self, alpha):
        super(LITnet, self).__init__()
        self.alpha = nn.Parameter(alpha*torch.ones(1, requires_grad=True))
    def forward(self, x):
        return LIT.apply(x, self.alpha)

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

class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        # self.middle_linear = torch.nn.Linear(H, H)
        self.lit1 = LITnet(10.0)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """

        h_relu = self.input_linear(x).clamp(min=0)

        # for _ in range(random.randint(0, 3)):
        #     h_relu = self.middle_linear(h_relu).clamp(min=0)

        # h_relu = self.middle_linear(h_relu).clamp(min=0)
        # h_relu = self.middle_linear(h_relu)
        h_relu = self.lit1(h_relu)
        # y_pred = self.lit1(x)
        y_pred = self.output_linear(h_relu)
        return y_pred


if __name__ == '__main__':

    dtype = torch.float
    # device = torch.device("cpu")
    device = torch.device("cuda") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    torch.manual_seed(1)

    # Create random Tensors to hold input and outputs.
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Construct our model by instantiating the class defined above
    model = DynamicNet(D_in, H, D_out).to(device)

    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.01)

    for t in range(200):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x).to(device)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(loss.item())
        # print ("\n\n")
        # print (model.lit1.alpha.grad)
        # print ("\n\n")

        # Zero gradients, perform a backward pass, and update the weights.
        # print (model.lit1.alpha.grad)
        optimizer.zero_grad()
        # print ("Zeroing gradients")
        # print (model.lit1.alpha.grad)
        loss.backward()
        optimizer.step()

        # Update weights using gradient descent
        # with torch.no_grad():
        #     w1 -= learning_rate * w1.grad
        #     w2 -= learning_rate * w2.grad

        #     # Manually zero the gradients after updating weights
        #     w1.grad.zero_()
        #     w2.grad.zero_()
