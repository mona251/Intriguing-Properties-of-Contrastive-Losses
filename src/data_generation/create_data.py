import numpy as np
import matplotlib.pyplot as plt
from src.utils import downsample_img, get_bottom_right_corner_to_match_shapes
from src.load_data import sample_uniformly_imgs
import os
import cv2 as cv
import random


def overlay_img(small_img, large_img, is_large_img_grayscale,
                bb_start_h, bb_end_h, bb_start_w, bb_end_w):
    """
    Overlays a small_img on a large_img at a specified bounding box.
    Args:
        small_img: small image
        large_img: large image
        is_large_img_grayscale: True if large_img is grayscale
        bb_start_h: y coordinate of the top corners of the bounding box
         inside large_img that will contain small_img
        bb_end_h: y coordinate of the bottom corners of the bounding box
         inside large_img that will contain small_img
        bb_start_w: x coordinate of the left corners of the bounding box
         inside large_img that will contain small_img
        bb_end_w: x coordinate of the right corners of the bounding box
         inside large_img that will contain small_img
    """
    if is_large_img_grayscale:
        large_img[bb_start_h:bb_end_h, bb_start_w:bb_end_w] += \
            small_img
        large_img[large_img > 255] = 255
    else:
        small_img_rgb = cv.cvtColor(small_img, cv.COLOR_GRAY2RGB)
        alpha_s = small_img_rgb[:, :, 2] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(large_img.shape[-1]):
            large_img[bb_start_h:bb_end_h, bb_start_w:bb_end_w, c] = \
                (alpha_s * small_img_rgb[:, :, c] +
                 alpha_l * large_img[bb_start_h:bb_end_h,
                 bb_start_w:bb_end_w, c])
    return large_img


def overlay_small_img_on_large_img_grid(small_img, large_img, num_cells_grid,
                                        n_repetitions_small_img,
                                        is_large_img_grayscale,
                                        repeat_small_img=True, dataset=None,
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
        repeat_small_img: True if the small_img should be repeated in the grid
        dataset: dataset that contain samples of small images. Set to None if
          repeat_small_img is True
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

    col_idx = 0
    row_idx = 0
    counter = 0

    final_img = np.copy(large_img)

    # random.seed(seed)
    sampled_idxs_of_cells = random.sample(range(num_cells_in_grid),
                                          n_repetitions_small_img)

    for i in range(num_cells_in_grid):
        if i in sampled_idxs_of_cells:
            cell_start_h = row_idx * int(height_size_cell)
            cell_start_w = col_idx * int(width_size_cell)

            cell_end_h = (row_idx + 1) * int(height_size_cell)
            cell_end_w = (col_idx + 1) * int(width_size_cell)
            if repeat_small_img:
                final_img = overlay_img(
                    small_img, final_img, is_large_img_grayscale, cell_start_h,
                    cell_end_h, cell_start_w, cell_end_w)
            else:
                # Sample a new image from the dataset for each cell of the grid
                # to fill
                small_img = sample_uniformly_imgs(dataset, 1)[0]
                small_img = downsample_img(small_img, height_size_cell,
                                           width_size_cell, grayscale=True)
                final_img = overlay_img(
                    small_img, final_img, is_large_img_grayscale, cell_start_h,
                    cell_end_h, cell_start_w, cell_end_w)
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


def overlay_small_img_on_large_img_at_random_position(large_img, small_img,
                                                      is_large_img_grayscale):
    """
    Overlays a small_img on a large_img in a random position.

    Args:
        large_img: big image that will contain small_img
        small_img: small image that will be added to big
        is_large_img_grayscale: True if large_img is grayscale

    """
    half_height_small_img = small_img.shape[0] // 2
    half_width_small_img = small_img.shape[1] // 2

    big_img_h, big_img_w = large_img.shape[:2]
    # range of coordinates of the portion of the large_img that can
    # contain the small_img
    top_left_coords_range = (half_height_small_img + 1, half_width_small_img + 1)
    bottom_right_coords_range = (big_img_h - half_height_small_img - 1,
                                 big_img_w - half_width_small_img - 1)
    # coordinates of where the center of the small_img will be
    # wrt the large_img
    center_y = np.random.randint(top_left_coords_range[0],
                                 bottom_right_coords_range[0])
    center_x = np.random.randint(top_left_coords_range[1],
                                 bottom_right_coords_range[1])
    # Coordinates of the portion of the large_img that will contain the
    # small_img (h, w)
    top_left_coord_small_img = (center_y - half_height_small_img,
                                center_x - half_width_small_img)
    bottom_right_coord_small_img = (center_y + half_height_small_img,
                                    center_x + half_width_small_img)
    # height of the bounding box that will contain the small image
    height_bb_small_img = \
        bottom_right_coord_small_img[0] - top_left_coord_small_img[0]
    width_bb_small_img = \
        bottom_right_coord_small_img[1] - top_left_coord_small_img[1]
    if height_bb_small_img != small_img.shape[0]:
        bottom_right_coord_small_img = \
            get_bottom_right_corner_to_match_shapes(
                height_bb_small_img, small_img.shape[0],
                bottom_right_coord_small_img, top_left_coord_small_img,
                update_height=True)

    if width_bb_small_img != small_img.shape[1]:
        bottom_right_coord_small_img = \
            get_bottom_right_corner_to_match_shapes(
                width_bb_small_img, small_img.shape[1],
                bottom_right_coord_small_img, top_left_coord_small_img,
                update_height=False)
    large_img = overlay_img(
        small_img, large_img, is_large_img_grayscale,
        top_left_coord_small_img[0], bottom_right_coord_small_img[0],
        top_left_coord_small_img[1], bottom_right_coord_small_img[1])

    return large_img
