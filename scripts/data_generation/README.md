# Generation of datasets

This folder contains Jupyter notebook files that create images similar to the datasets used in the original [paper](https://proceedings.neurips.cc/paper/2021/hash/628f16b29939d1b060af49f66ae0f7f8-Abstract.html).

In particular:
- `multi_digits_grid.ipynb` generates images of MultiDigits dataset, with digits placed on a grid.
- `multi_digits_randomly_placed.ipynb` generates images of MultiDigits dataset, with digits placed randomly.
- `multi_digits_varying_size_randomly_placed.ipynb` generates images of MultiDigits dataset, with two digits placed randomly. One has a fixed size, while the size of other one varies.
- `digits_on_another_dataset_grid.ipynb` generates images of DigitOnImageNet dataset. For each ImageNet image, a unique MNIST digit is replicated
on a grid. Other datasets could be used instead of ImageNet.