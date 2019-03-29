import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == '__main__':

    data = np.loadtxt("results/resnet20epochs200.txt", delimiter=',')
    data2 = np.loadtxt("results/litresnet20epochs200.txt", delimiter=',')

    # print(data.shape)

    losses_train = data[:,0]
    losses_train2 = data2[:,0]

    # accuracy_test = data[:,1]

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title('Training Error')
    ax.set_yscale("log", nonposy='clip')
    ax.plot(losses_train, 'b-', label="Full Resolution")
    ax.plot(losses_train2, 'r--', label="4-bit LIT")    
    ax.set_ylabel('Loss') #, color='b')
    ax.set_xlabel('Epoch')
    ax.legend()

    # ax2 = ax.twinx()
    # ax2.plot(accuracy_test, 'r-')
    # ax2.set_ylabel('Accuracy (%)', color = 'r')
    
    plt.show()