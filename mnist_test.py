from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os

class LIT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DumbNet(nn.Module):
    def __init__(self):
        super(DumbNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = LIT.apply(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = LIT.apply(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = LIT.apply(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LitNet(nn.Module):
    def __init__(self):
        super(LitNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    # model.train()
    epoch_loss = 0.0
    batch_loss = 0.0
    num_passes = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        epoch_loss += loss.item()
        num_passes += 1.0
        if batch_idx % args.log_interval == args.log_interval-1:
            print('[{}, {}] loss: {:.6f}'.format(
                epoch, batch_idx, loss.item()))
            batch_loss = 0.0
    return epoch_loss/float(num_passes)

def test(args, model, device, test_loader):
    # model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            images, labels = inputs.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print('\nTest set accuracy: {:.0f}%\n'.format(
        # 100 * correct / total ))
    return 100 * correct / total

def main():
    start = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--file-name', type=str, default='test_' + str(int(start))[-3:], metavar='filename',
                        help='Name of file to store model and losses')
    parser.add_argument('--quant-type', type=str, default='none', metavar='qtype',
                        help='Type of quantisation used on activation functions')
    parser.add_argument('--bit-res', type=int, default=4, metavar='bitres',
                        help='Bit resolution of activation funtion')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    qt = args.quant_type
    if qt == 'dumb':
        model = DumbNet().to(device)
        print("Building dumb {0} bit network".format(args.bit_res))
    elif qt == 'lit':
        model = LitNet().to(device)
        print("Building LIT {0} bit network".format(args.bit_res))
    else:
        model = Net().to(device)
        print("\nBuilding full resolution network")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    losses_train = np.zeros((args.epochs))
    accuracy_test  = np.zeros((args.epochs))

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_train_loss    = train(args, model, device, train_loader, optimizer, epoch)
        epoch_test_accuracy = test(args, model, device, test_loader)
        losses_train[epoch-1] = epoch_train_loss 
        accuracy_test[epoch-1]  = epoch_test_accuracy
        current_time = time.time() - start
        print('\nEpoch {:d} summary'.format(epoch))
        print('Training set average loss: {:.6f}'.format(epoch_train_loss))
        print('Test set accuracy: {:.0f}%'.format(epoch_test_accuracy))
        # print('Test set loss: {:.6f}'.format(epoch_test_accuracy))
        print('Time taken: {:.3f}s\n'.format(current_time))

    if (args.save_model):
        if not os.path.exists('models'):
            os.mkdir('models')
        torch.save(model.state_dict(),'models/' + args.file_name+'.pt')
        if not os.path.exists('results'):
            os.mkdir('results')
        losses = np.stack((losses_train, accuracy_test), axis=1)
        np.savetxt('results/' + args.file_name+'.txt', losses, delimiter=', ')

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title('Training Loss / Testing Accuracy')
    ax.plot(losses_train, 'b-')
    ax.set_ylabel('Loss', color='b')
    ax.set_xlabel('Epoch')

    ax2 = ax.twinx()
    ax2.plot(accuracy_test, 'r-')
    ax2.set_ylabel('Accuracy (%)', color = 'r')
    
    # blue_line = mpatches.Patch(color='blue', label='Training Loss')
    # orange_line = mpatches.Patch(color='orange', label='Testing Accuracy')
    # plt.legend(handles=[blue_line, orange_line])
    plt.show()
        
if __name__ == '__main__':
    main()