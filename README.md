# DD2412project

Pre-trained simCLR: https://github.com/Spijkervet/SimCLR

Experimental setup:

ResNet-50 with 2-layer projection head after the ResNet's average pooling layer.
Projection dim = 64 for CIFAR10 128 for ImageNet
LearningRate = 0.075 * sqrt(BatchSize) for ImageNet 0.2 * sqrt(BatchSize) for CIFAR10

