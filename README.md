# DD2412project


[![Open in Colab] (https://colab.research.google.com/assets/colab-badge.svg)] (https://colab.research.google.com/github/mona251/DD2412project/blob/main/simclr.ipynb)

Pre-trained simCLR: https://github.com/Spijkervet/SimCLR

Download imagenette here (click Full Size download): https://github.com/fastai/imagenette

Experimental setup:

ResNet-50 with 2-layer projection head after the ResNet's average pooling layer.

Projection dim = 64 for CIFAR10 128 for ImageNet

LearningRate = 0.075 * sqrt(BatchSize) for ImageNet 0.2 * sqrt(BatchSize) for CIFAR10

NT-Xent loss temperature = 0.2

Decoupled NT-Xent loss temperature = 1.0, lambda = 0.1

SWD based losses lambda = 5

BatchSize = 128 for CIFAR10 1024 for ImageNet
