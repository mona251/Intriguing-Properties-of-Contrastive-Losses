from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np


def load_mnist():
    mnist_dataset_train = MNIST('../data/mnist/', download=True)
    mnist_dataset_test = MNIST('../data/mnist/', download=True, train=False)
    return mnist_dataset_train, mnist_dataset_test


def get_ith_img(dataset, idx, plot=False):
    """
    Gets the image that corresponds to index idx in dataset.
    Args:
        dataset: dataset
        idx: index of the image to get
        plot: True to show the image

    Returns:
        The image that corresponds to index idx in dataset.
    """
    x, _ = dataset[idx]
    img = np.asarray(x)
    if plot:
        plt.imshow(img, cmap='gray_r')
        plt.show()
    return img


def sample_uniformly_img(dataset, seed):
    """
    Samples an image from dataset with discrete uniform probability.
    Args:
        dataset: dataset
        seed: seed

    Returns:
        The sampled image.
    """
    np.random.seed(seed)
    num_samples_in_dataset = len(dataset)
    sampled_idx = np.random.randint(0, num_samples_in_dataset)
    sampled_img, _ = dataset[sampled_idx]
    return np.asarray(sampled_img)
