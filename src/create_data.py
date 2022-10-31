import numpy as np
import matplotlib.pyplot as plt
from src.utils import downsample_img
import os
import cv2 as cv
import random


def overlay_small_img_on_large_img_grid(small_img, large_img,
                                        num_cells_grid,
                                        n_repetitions_small_img,
                                        is_large_img_grayscale=False,
                                        plot=False):
    """
    Returns a grid of n repetitions of a small image above a large_img.
    Args:
        small_img: a small image
        large_img: a large image
        num_cells_grid: number of cells in the grid
        n_repetitions_small_img: n repetitions of small_image (number of cells
         that will contain small_img)
        is_large_img_grayscale: True if large_img is a grayscale image
        plot: True to see the plot of the grid

    Returns:
        A copy of the large image with a grid of n repetitions of a small image
         above it.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    num_cells_in_grid = num_cells_grid
    num_cells_one_row = np.sqrt(num_cells_in_grid)
    num_cells_one_col = num_cells_one_row

    height_big, width_big = large_img.shape[:2]
    height_size_cell = height_big // num_cells_one_col
    width_size_cell = width_big // num_cells_one_row

    small_img = downsample_img(small_img, height_size_cell, width_size_cell,
                               grayscale=True)
    if not is_large_img_grayscale:
        backtorgb = cv.cvtColor(small_img, cv.COLOR_GRAY2RGB)

    col_idx = 0
    row_idx = 0
    counter = 0

    final_img = np.copy(large_img)

    #random.seed(seed)
    sampled_idxs_of_cells = random.sample(range(num_cells_in_grid),
                                          n_repetitions_small_img)

    for i in range(num_cells_in_grid):
        if i in sampled_idxs_of_cells:
            cell_start_h = row_idx * int(height_size_cell)
            cell_start_w = col_idx * int(width_size_cell)

            cell_end_h = (row_idx + 1) * int(height_size_cell)
            cell_end_w = (col_idx + 1) * int(width_size_cell)

            if is_large_img_grayscale:
                final_img[cell_start_h:cell_end_h, cell_start_w:cell_end_w] = \
                    (small_img + final_img[cell_start_h:cell_end_h,
                                           cell_start_w:cell_end_w])
            else:
                alpha_s = backtorgb[:, :, 2] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(final_img.shape[-1]):
                    final_img[cell_start_h:cell_end_h, cell_start_w:cell_end_w, c] = \
                        (alpha_s * backtorgb[:, :, c] +
                         alpha_l * final_img[cell_start_h:cell_end_h,
                                             cell_start_w:cell_end_w, c])
        counter += 1
        col_idx += 1
        if counter % num_cells_one_row == 0:
            row_idx += 1
            counter = 0
            col_idx = 0
    if not is_large_img_grayscale:
        final_img = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
    final_img = np.array(final_img)
    if plot:
        plt.imshow(final_img)
        plt.show()

    return final_img
