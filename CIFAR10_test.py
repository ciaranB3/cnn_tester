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

from resnet_model import *

    
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
    file_name = "ResNet20_CIFAR10"
    network = "full20"
    bit_res = 7

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

    fig = plt.figure(1)
    ax = fig.gca()
    ax.set_title('Full Resolution ResNet-20')
    ax.plot(losses_train, 'b-')
    ax.set_ylabel('Loss', color='b')
    ax.set_xlabel('Epoch')

    ax2 = ax.twinx()
    ax2.plot(accuracy_test, 'r-')
    ax2.set_ylabel('Accuracy (%)', color = 'r')

    # # Training settings

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
    # file_name = "LitResNet20_CIFAR10"
    # network = "lit20"
    # bit_res = 4

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

    # fig2 = plt.figure(2)
    # ax = fig2.gca()
    # ax.set_title('4-bit Activations ResNet-20')
    # ax.plot(losses_train, 'b-')
    # ax.set_ylabel('Loss', color='b')
    # ax.set_xlabel('Epoch')

    # ax2 = ax.twinx()
    # ax2.plot(accuracy_test, 'r-')
    # ax2.set_ylabel('Accuracy (%)', color = 'r')
    
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