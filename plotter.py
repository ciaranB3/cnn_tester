import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == '__main__':

    data = np.loadtxt("results/ResNet20_CIFAR10.txt", delimiter=',')
    data2 = np.loadtxt("results/LitResNet20_CIFAR10.txt", delimiter=',')

    # print(data.shape)

    losses_train = data[:,0]
    losses_train2 = data2[:,0]


    accuracy_test = data[:,1]
    accuracy_test2 = data2[:,1]


    fig = plt.figure(1)
    ax = fig.gca()
    ax.set_title('Training Loss')
    ax.set_yscale("log", nonposy='clip')
    ax.plot(losses_train, 'b-', label="Full Resolution")
    ax.plot(losses_train2, 'r--', label="4-bit PACT")    
    ax.set_ylabel('Cross Entropy Loss') #, color='b')
    ax.set_xlabel('Epoch')
    ax.legend()

    fig2 = plt.figure(2)
    ax2 = fig2.gca()
    ax2.set_title('Validation Accuracy')
    # ax2.set_yscale("log", nonposy='clip')
    ax2.plot(accuracy_test, 'b-', label="Full Resolution")
    ax2.plot(accuracy_test2, 'r--', label="4-bit PACT")    
    ax2.set_ylabel('Accuracy (%)') #, color='b')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    print(np.amax(accuracy_test))
    print(np.amax(accuracy_test2))

    # ax2 = ax.twinx()
    # ax2.plot(accuracy_test, 'r-')
    # ax2.set_ylabel('Accuracy (%)', color = 'r')
    
    plt.show()