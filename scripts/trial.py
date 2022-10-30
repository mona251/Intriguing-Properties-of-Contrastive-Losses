import numpy as np
import matplotlib.pyplot as plt
from src.load_data import load_mnist, get_ith_img
from src.utils import downsample_img
import os


def other_main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    mnist_dataset_train, mnist_dataset_test = load_mnist()

    # TODO: sample randomly a digit img
    # TODO: what about the dim of the mnist img? Is it fine to just
    #  downsample/upsample it to the dimension of a cell of a grid?

    img_big = np.zeros((200, 200))

    num_cells_in_grid = 16
    num_cells_one_row = np.sqrt(num_cells_in_grid)
    num_cells_one_col = num_cells_one_row

    height_big, width_big = img_big.shape
    height_size_cell = height_big / num_cells_one_col
    width_size_cell = width_big / num_cells_one_row

    digit_img = get_ith_img(mnist_dataset_train, 0, plot=True)
    print(digit_img.shape)
    digit_img = downsample_img(digit_img, height_size_cell, width_size_cell,
                               grayscale=True)
    print(digit_img.shape)

    big_img = np.zeros((200, 200))

    col_idx = 0
    row_idx = 0
    counter = 0

    for i in range(num_cells_in_grid):
        print(row_idx, col_idx)

        cell_start_h = row_idx * int(height_size_cell)
        cell_start_w = col_idx * int(width_size_cell)

        cell_end_h = (row_idx + 1) * int(height_size_cell)
        cell_end_w = (col_idx + 1) * int(width_size_cell)

        big_img[cell_start_h:cell_end_h, cell_start_w:cell_end_w] = digit_img
        counter += 1
        col_idx += 1
        if counter % num_cells_one_row == 0:
            row_idx += 1
            counter = 0
            col_idx = 0

    plt.imshow(big_img)
    plt.show()

def main():
    img_big = np.zeros((200, 200))

    num_cells_in_grid = 16
    num_cells_one_row = np.sqrt(num_cells_in_grid)
    num_cells_one_col = num_cells_one_row

    height_big, width_big = img_big.shape
    height_size_cell = height_big / num_cells_one_row
    width_size_cell = width_big / num_cells_one_col

    big_img = np.zeros((200, 200))

    for i in range(num_cells_one_row):
        for j in range(num_cells_one_col):

            print(i, j)

            cell_start_h = i * int(height_size_cell)
            cell_start_w = j * int(width_size_cell)

            cell_end_h = (i + 1) * int(height_size_cell)
            cell_end_w = (j + 1) * int(width_size_cell)

            big_img[cell_start_h:cell_end_h, cell_start_w:cell_end_w] = 255

    print(np.unique(big_img))
    plt.imshow(big_img)
    plt.show()


if __name__ == "__main__":
    print(np.__version__)
    other_main()
