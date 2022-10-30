from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np


def load_mnist():
    mnist_dataset_train = MNIST('../data/mnist/', download=True)
    mnist_dataset_test = MNIST('../data/mnist/', download=True, train=False)
    return mnist_dataset_train, mnist_dataset_test


def show_ith_image(dataset, i):
    x, _ = dataset[i]
    plt.imshow(np.asarray(x), cmap='gray_r')
