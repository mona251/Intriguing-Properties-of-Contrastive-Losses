from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np


def load_mnist():
    mnist_dataset_train = MNIST('../data/mnist/', download=True)
    mnist_dataset_test = MNIST('../data/mnist/', download=True, train=False)
    return mnist_dataset_train, mnist_dataset_test


def get_ith_img(dataset, idx, plot=False):
    x, _ = dataset[idx]
    img = np.asarray(x)
    if plot:
        plt.imshow(img, cmap='gray_r')
        plt.show()
    return img

