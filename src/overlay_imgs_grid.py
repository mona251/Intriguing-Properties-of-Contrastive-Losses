import numpy as np
import matplotlib.pyplot as plt
from src.load_data import load_mnist, get_ith_img
from src.utils import downsample_img
import os
import cv2 as cv


def overlay_small_img_on_large_img_grid(small_img, large_img,
                                        n_repetitions_small_img):
    """
    Returns a grid of n repetitions of a small image above a large_img.
    Args:
        small_img: a small image
        large_img: a large image
        n_repetitions_small_img: n repetitions of small_image

    Returns:
        The large image with a grid of n repetitions of a small image above it.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    num_cells_in_grid = n_repetitions_small_img
    num_cells_one_row = np.sqrt(num_cells_in_grid)
    num_cells_one_col = num_cells_one_row

    height_big, width_big = large_img.shape[:2]
    height_size_cell = height_big / num_cells_one_col
    width_size_cell = width_big / num_cells_one_row

    small_img = downsample_img(small_img, height_size_cell, width_size_cell,
                               grayscale=True)
    backtorgb = cv.cvtColor(small_img, cv.COLOR_GRAY2RGB)

    col_idx = 0
    row_idx = 0
    counter = 0

    for i in range(num_cells_in_grid):
        cell_start_h = row_idx * int(height_size_cell)
        cell_start_w = col_idx * int(width_size_cell)

        cell_end_h = (row_idx + 1) * int(height_size_cell)
        cell_end_w = (col_idx + 1) * int(width_size_cell)

        alpha_s = backtorgb[:, :, 2] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(large_img.shape[-1]):
            large_img[cell_start_h:cell_end_h, cell_start_w:cell_end_w, c] = \
                (alpha_s * backtorgb[:, :, c] +
                 alpha_l * large_img[cell_start_h:cell_end_h,
                                     cell_start_w:cell_end_w, c])
        counter += 1
        col_idx += 1
        if counter % num_cells_one_row == 0:
            row_idx += 1
            counter = 0
            col_idx = 0

    large_img = cv.cvtColor(large_img, cv.COLOR_BGR2RGB)
    large_img = np.array(large_img)
    plt.imshow(large_img)
    plt.show()


def main():
    n_repetitions_small_img = 9
    large_img_path = "../data/img_trial/abc.png"
    large_img = cv.imread(large_img_path)
    mnist_dataset_train, mnist_dataset_test = load_mnist()
    # TODO: sample randomly a digit img
    # TODO: what about the dim of the mnist img? Is it fine to just
    #  downsample/upsample it to the dimension of a cell of a grid?
    small_img = get_ith_img(mnist_dataset_train, 0, plot=False)
    overlay_small_img_on_large_img_grid(small_img, large_img,
                                        n_repetitions_small_img)


if __name__ == "__main__":
    main()
