import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_height_width_ratio(old_height: int, old_width: int, new_height: int,
                           new_width: int) -> (float, float):
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


def downsample_img(original_img: np.ndarray, new_height: int, new_width: int,
                   grayscale: bool, interpolation_method=cv.INTER_LINEAR) \
        -> np.ndarray:
    """Downsamples an image.
    Args:
        original_img (numpy.array): original image
        new_height (int): height that the image will have
        new_width (int): width that the image will have
        grayscale (bool): True if original_img is a grayscale image
        interpolation_method: interpolation method
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
                                interpolation=interpolation_method)
    return downsampled_img


def crop_digit(img: np.ndarray, plot=False) -> np.ndarray:
    """
    Given an image with a black background, it returns the smallest bounding
    box that contains the pixels that are both black and not black inside
    the image img.
    (For instance, given an image of a digit with a big portion of black
    background, it returns the image of the digit inside the bounding box
    that contains all the pixels of the digit and the smallest portion of
    black pixels).
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
        plt.show()

    return digit_cropped


def get_bottom_right_corner_to_match_shapes(shape_1: int, shape_2: int,
                                            bottom_right_corner: (int, int),
                                            top_left_corner: (int, int),
                                            update_height: bool) -> (int, int):
    """
    Updates the bottom right corner coordinates such that shape_1 and shape_2
    will have the same length.
    Args:
        shape_1: first shape
        shape_2: second shape
        bottom_right_corner: coordinates of the bottom right corner of a
         bounding box
        top_left_corner: coordinates of the top left corner of a
         bounding box
        update_height: True if shape_1 and shape_2 are heights, False if they
         are widths

    Returns:
        The updated coordinates of the bottom right corner.
    """
    i = 0
    while shape_1 < shape_2:
        i += 1
        if update_height:
            bottom_right_corner = (bottom_right_corner[0] + i,
                                   bottom_right_corner[1])
            shape_1 = \
                bottom_right_corner[0] - top_left_corner[0]
        else:
            bottom_right_corner = (bottom_right_corner[0],
                                   bottom_right_corner[1] + i)
            shape_1 = \
                bottom_right_corner[1] - top_left_corner[1]
    while shape_1 > shape_2:
        i += 1
        if update_height:
            bottom_right_corner = (bottom_right_corner[0] - i,
                                   bottom_right_corner[1])
            shape_1 = \
                bottom_right_corner[0] - top_left_corner[0]
        else:
            bottom_right_corner = (bottom_right_corner[0],
                                   bottom_right_corner[1] - i)
            shape_1 = \
                bottom_right_corner[1] - top_left_corner[1]

    return bottom_right_corner


def normalize_img(img: np.ndarray, min_value: int, max_value: int,
                  return_int_values=False) -> np.ndarray:
    """
    Normalizes an image by changing its pixels' values range into [min_value,
    max_value].
    Args:
        img: image
        min_value: minimum value of the range of values pixels will be scaled
         to
        max_value: maximum value of the range of values pixels will be scaled
         to
        return_int_values: True if the normalized values should be integers

    Returns:
        The normalized image.
    """
    norm_image = cv.normalize(img, None, alpha=min_value, beta=max_value,
                              norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    if return_int_values:
        # To use for example when normalizing an image between 0 and 255.
        # Not to use when normalizing an image between [0, 1(.
        norm_image = norm_image.astype(np.uint8)
    return norm_image
