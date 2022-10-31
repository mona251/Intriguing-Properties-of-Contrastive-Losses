import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_height_width_ratio(old_height, old_width, new_height, new_width):
    """Returns the two ratios:
      - new height / old height
      - new witdh / old witdh
    Args:
        old_height (int): height that the image had
        old_width (int): width that the image had
        new_height (int): height that the image will have
        new_width (int): width that the image will have
    Returns:
        (float, float): two ratios.
    """
    height_ratio = new_height / old_height
    width_ratio = new_width / old_width
    return height_ratio, width_ratio


def downsample_img(original_img, new_height, new_width, grayscale):
    """Downsamples an image.
    Args:
        original_img (numpy.array): original image
        new_height (int): height that the image will have
        new_width (int): width that the image will have
        grayscale (bool): True if original_img is a grayscale image
    Returns:
        numpy.array: downsampled image.
    """
    if grayscale:
        old_height, old_width = original_img.shape
    else:
        old_height, old_width, _ = original_img.shape
    height_ratio, width_ratio = get_height_width_ratio(old_height, old_width,
                                                       new_height, new_width)

    downsampled_img = cv.resize(original_img,  # original image
                                (0, 0),  # set fx and fy, not the final size
                                fx=width_ratio,
                                fy=height_ratio,
                                interpolation=cv.INTER_LINEAR)
    return downsampled_img


def crop_digit(img, plot=False):
    """
    Given an image with a digit, it returns the smallest bounding box that
    contains the digit.
    Args:
        img: Image that contains the digit
        plot: True to show the image

    Returns:
        The smallest bounding box in img that contains the digit.
    """
    background_color = 0
    not_black_pixels_coords = np.where(img > background_color)
    # top-most y coord of not black (not background) pixel
    top_most_not_black_y = np.min(not_black_pixels_coords[0])
    # bottom-most y coord of not black (not background) pixel
    bottom_most_not_black_y = np.max(not_black_pixels_coords[0])
    # left-most x coord of not black (not background) pixel
    left_most_not_black_x = np.min(not_black_pixels_coords[1])
    # right-most x coord of not black (not background) pixel
    right_most_not_black_x = np.max(not_black_pixels_coords[1])

    digit_cropped = img[top_most_not_black_y:bottom_most_not_black_y + 1,
                        left_most_not_black_x:right_most_not_black_x + 1]
    if plot:
        plt.imshow(digit_cropped)
