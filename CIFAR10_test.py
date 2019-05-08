from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision import datasets, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import os

class LIT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=10.0*torch.autograd.Variable(torch.ones(1,1)), num_bits=4):
        # x, alpha = ctx.saved_tensors
        # print("Forward alpha: {}".format(alpha.data))
        # print("Max input value: {}".format(x.max()))
        # if num_bits!=4:
            # print("num_bits = {}".format(num_bits))
        ctx.num_bits = num_bits
        ctx.save_for_backward(x, alpha)
        output = x.clamp(min=0, max=alpha.data[0])
        scale = ((2**num_bits)-1)/alpha
        output = torch.round(output * scale) / scale
        # output = x.clamp(min=0)
        return output

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


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', position='none', bit_res=4):
        super(BasicBlock, self).__init__()
        self.position = position
        self.bit_res = bit_res
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if position!='none':
            self.lit1 = LITnet(alpha=10.0, bit_res=bit_res)
            if position=='middle':
                self.lit2 = LITnet(alpha=10.0, bit_res=bit_res)
            else:
                 self.lit2 = LITnet(alpha=10.0, bit_res=2*bit_res)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        if self.position=='first':
            out = self.lit2(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.lit1(out)
        elif self.position=='middle':
            out = self.lit1(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.lit2(out)
        elif self.position=='last':
            out = self.lit1(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.lit2(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, quant='none', bit_res=4):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.quant = quant
        self.bit_res = bit_res

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        if self.quant=='none':
            self.layer1 = self._make_layer(block, 16, 1, stride=1)
            self.layer2 = self._make_layer(block, 16, num_blocks[0]-1, stride=1)
            self.layer3 = self._make_layer(block, 32, num_blocks[1], stride=2)
            self.layer4 = self._make_layer(block, 64, num_blocks[2]-1, stride=2)
            self.layer5 = self._make_layer(block, 64, 1, stride=1)
        else:
            self.layer1 = self._make_layer(block, 16, 1, stride=1, position='first', bit_res=self.bit_res)
            self.layer2 = self._make_layer(block, 16, num_blocks[0]-1, stride=1, position='middle', bit_res=self.bit_res)
            self.layer3 = self._make_layer(block, 32, num_blocks[1], stride=2, position='middle', bit_res=self.bit_res)
            self.layer4 = self._make_layer(block, 64, num_blocks[2]-1, stride=2, position='middle', bit_res=self.bit_res)
            self.layer5 = self._make_layer(block, 64, 1, stride=1, position='last', bit_res=self.bit_res)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, position='none', bit_res=4):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, position=position, bit_res=bit_res))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class LitResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bit_res=4):
        super(LitResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, 1, stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[0]-1, stride=1)
        self.layer3 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[2]-1, stride=2)
        self.layer5 = self._make_layer(block, 64, 1, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 3, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def litresnet20(bit_res=4, pretrained=False):
    """Constructs a ResNet-20 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 3, 3], quant='lit', bit_res=bit_res)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
    
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    epoch_loss = 0.0
    batch_loss = 0.0
    num_passes = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        epoch_loss += loss.item()
        num_passes += 1.0
        if batch_idx % log_interval == log_interval-1:
            print('[{}, {}] loss: {:.6f}'.format(
                epoch, batch_idx, loss.item()))
            batch_loss = 0.0
    return epoch_loss/float(num_passes)

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            images, labels = inputs.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def main():
    ## Training settings

    # batch_size = 128
    # test_batch_size = 1000
    # epochs = 200
    # learning_rate = 0.1
    # momentum = 0.9
    # weight_decay = 0.0002
    # no_cuda = False
    # seed = 1
    # log_interval = 2000
    # save_model = True 
    # file_name = "ResNet20_CIFAR10"
    # network = "full20"
    # bit_res = 7

    # use_cuda = not no_cuda and torch.cuda.is_available()
    # print(torch.cuda.is_available())
    # torch.manual_seed(seed)
    # device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    # transform = transforms.Compose(
    # [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomCrop(32, 4),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]), download=True),
    #     batch_size=batch_size, shuffle=True,
    #     num_workers=2, pin_memory=True)

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=128, shuffle=False,
    #     num_workers=2, pin_memory=True)

    # if network == 'lit':
    #     model = litresnet18().to(device)
    #     print("Building LIT {0} bit resnet-18".format(bit_res))
    # elif network == 'lit20':
    #     model = litresnet20(bit_res=bit_res).to(device)
    #     print("Building LIT {0} bit ResNet-20".format(bit_res))
    # elif network == 'full20':
    #     model = resnet20().to(device)
    #     print("Building full resolution ResNet-20")
    # else:
    #     model = resnet18().to(device)
    #     print("\nBuilding full resolution ResNet-18")

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # losses_train = np.zeros((epochs))
    # accuracy_test  = np.zeros((epochs))

    # print(model.parameters)

    # start = time.time()

    # for epoch in range(0, epochs):
    #     if epoch==60:
    #         learning_rate = learning_rate/10.0
    #         print("\nUpdating learning rate to {}\n".format(learning_rate))
    #         optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
    #             momentum=momentum, weight_decay=0.0002)
    #     if epoch==120:
    #         learning_rate = learning_rate/10.0
    #         print("\nUpdating learning rate to {}\n".format(learning_rate))
    #         optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
    #             momentum=momentum, weight_decay=0.0002)
    #     epoch_train_loss    = train(model, device, train_loader, optimizer, epoch, log_interval)
    #     epoch_test_accuracy = test(model, device, test_loader)
    #     losses_train[epoch] = epoch_train_loss 
    #     accuracy_test[epoch]  = epoch_test_accuracy
    #     current_time = time.time() - start
    #     print('\nEpoch {:d} summary'.format(epoch))
    #     print('Training set average loss: {:.6f}'.format(epoch_train_loss))
    #     print('Test set accuracy: {:.2f}%'.format(epoch_test_accuracy))
    #     # print('Layer1 alpha1: {} Layer1 alpha2: {}'.format(
    #         # model.layer1[0].lit1.alpha.data[0].item(), model.layer1[0].lit2.alpha.data[0].item()))
    #     print('Time taken: {:.3f}s\n'.format(current_time))

    # if (save_model):
    #     if not os.path.exists('models'):
    #         os.mkdir('models')
    #     torch.save(model.state_dict(),'models/' + file_name+'.pt')
    #     if not os.path.exists('results'):
    #         os.mkdir('results')
    #     losses = np.stack((losses_train, accuracy_test), axis=1)
    #     np.savetxt('results/' + file_name+'.txt', losses, delimiter=', ')

    # fig = plt.figure(1)
    # ax = fig.gca()
    # ax.set_title('Full Resolution ResNet-20')
    # ax.plot(losses_train, 'b-')
    # ax.set_ylabel('Loss', color='b')
    # ax.set_xlabel('Epoch')

    # ax2 = ax.twinx()
    # ax2.plot(accuracy_test, 'r-')
    # ax2.set_ylabel('Accuracy (%)', color = 'r')

    # Training settings

    batch_size = 128
    test_batch_size = 1000
    epochs = 200
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 0.0002
    no_cuda = False
    seed = 1
    log_interval = 2000
    save_model = True 
    file_name = "LitResNet20_CIFAR10"
    network = "lit20"
    bit_res = 4

    use_cuda = not no_cuda and torch.cuda.is_available()
    print(torch.cuda.is_available())
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=2, pin_memory=True)

    if network == 'lit':
        model = litresnet18().to(device)
        print("Building LIT {0} bit resnet-18".format(bit_res))
    elif network == 'lit20':
        model = litresnet20(bit_res=bit_res).to(device)
        print("Building LIT {0} bit ResNet-20".format(bit_res))
    elif network == 'full20':
        model = resnet20().to(device)
        print("Building full resolution ResNet-20")
    else:
        model = resnet18().to(device)
        print("\nBuilding full resolution ResNet-18")

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    losses_train = np.zeros((epochs))
    accuracy_test  = np.zeros((epochs))

    print(model.parameters)

    start = time.time()

    for epoch in range(0, epochs):
        if epoch==60:
            learning_rate = learning_rate/10.0
            print("\nUpdating learning rate to {}\n".format(learning_rate))
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                momentum=momentum, weight_decay=0.0002)
        if epoch==120:
            learning_rate = learning_rate/10.0
            print("\nUpdating learning rate to {}\n".format(learning_rate))
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                momentum=momentum, weight_decay=0.0002)
        epoch_train_loss    = train(model, device, train_loader, optimizer, epoch, log_interval)
        epoch_test_accuracy = test(model, device, test_loader)
        losses_train[epoch] = epoch_train_loss 
        accuracy_test[epoch]  = epoch_test_accuracy
        current_time = time.time() - start
        print('\nEpoch {:d} summary'.format(epoch))
        print('Training set average loss: {:.6f}'.format(epoch_train_loss))
        print('Test set accuracy: {:.2f}%'.format(epoch_test_accuracy))
        # print('Layer1 alpha1: {} Layer1 alpha2: {}'.format(
            # model.layer1[0].lit1.alpha.data[0].item(), model.layer1[0].lit2.alpha.data[0].item()))
        print('Time taken: {:.3f}s\n'.format(current_time))

    if (save_model):
        if not os.path.exists('models'):
            os.mkdir('models')
        torch.save(model.state_dict(),'models/' + file_name+'.pt')
        if not os.path.exists('results'):
            os.mkdir('results')
        losses = np.stack((losses_train, accuracy_test), axis=1)
        np.savetxt('results/' + file_name+'.txt', losses, delimiter=', ')

    fig2 = plt.figure(2)
    ax = fig2.gca()
    ax.set_title('4-bit Activations ResNet-20')
    ax.plot(losses_train, 'b-')
    ax.set_ylabel('Loss', color='b')
    ax.set_xlabel('Epoch')

    ax2 = ax.twinx()
    ax2.plot(accuracy_test, 'r-')
    ax2.set_ylabel('Accuracy (%)', color = 'r')
    
    plt.show()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def quick_test():
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    model = litresnet18()
    model.load_state_dict(torch.load("models/litresnet18epochs200v3.pt"))
    model.eval()
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    for i in range(32):
        print('GroundTruth: {} \t Predicted: {}'.format(classes[labels[i]], classes[predicted[i]]))
    imshow(utils.make_grid(images))

def accuracy_test():
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    testset = datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False, num_workers=2)

    device = torch.device("cuda")
    model = resnet20()
    model.load_state_dict(torch.load("models/resnet20epochs200.pt"))
    model.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            images, labels = inputs.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

if __name__ == '__main__':
    main()
    # quick_test()
    # print(accuracy_test())