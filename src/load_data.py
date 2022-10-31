from torchvision.datasets import MNIST, ImageNet
import matplotlib.pyplot as plt
import numpy as np
import random


def load_mnist():
    mnist_dataset_train = MNIST('../data/mnist/', download=True)
    mnist_dataset_test = MNIST('../data/mnist/', download=True, train=False)
    return mnist_dataset_train, mnist_dataset_test


#def load_imagenet():
#    imagenet_dataset_train = ImageNet('../data/imagenet/', download=True)
#    imagenet_dataset_test = ImageNet('../data/imagenet/', download=True, train=False)
#    return imagenet_dataset_train, imagenet_dataset_test


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


def sample_uniformly_imgs(dataset, num_samples, seed):
    """
    Samples images (without duplicates) from dataset with discrete uniform
    probability.
    Args:
        dataset: dataset
        num_samples: number of images to sample
        seed: seed

    Returns:
        The sampled images.
    """
    random.seed(seed)
    sampled_imgs = []
    sampled_idxs = random.sample(range(len(dataset)), num_samples)
    
    for idx in sampled_idxs:
        img, _ = dataset[idx]
        sampled_imgs.append(np.asarray(img))
    return sampled_imgs
