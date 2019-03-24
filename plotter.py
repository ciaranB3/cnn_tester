import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = np.loadtxt("results/resnet18epochs200v2.txt", delimiter=',')
    # print(data.shape)

    losses_train = data[:,0]
    # accuracy_test = data[:,1]

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title('Training Error')
    ax.set_yscale("log", nonposy='clip')
    ax.plot(losses_train, 'b-')
    ax.set_ylabel('Loss') #, color='b')
    ax.set_xlabel('Epoch')

    # ax2 = ax.twinx()
    # ax2.plot(accuracy_test, 'r-')
    # ax2.set_ylabel('Accuracy (%)', color = 'r')
    
    plt.show()