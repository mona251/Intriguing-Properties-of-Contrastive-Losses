import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from src.data_generation.utils import downsample_img


def get_colors_of_clusters(n_clusters):
    """
    Gets the colors of each cluster for K-Means.
    Args:
        n_clusters: number of clusters

    Returns:
        The colors of each cluster for K-Means.
    """
    blue = [0, 76, 153]
    orange = [255, 178, 102]
    red = [255, 102, 102]
    light_blue = [102, 255, 255]
    green = [0, 153, 76]
    yellow = [255, 255, 153]
    violet = [204, 153, 255]
    pink = [255, 153, 153]
    black = [0, 0, 0]
    white = [255, 255, 255]
    colors = [blue, orange, red, light_blue, green, yellow, violet, pink,
              black, white]
    centers = []
    for i in range(n_clusters):
        centers.append(colors[i])
    centers = np.uint8(centers)

    return centers


def k_means_on_img(image, k, max_iter=100, epsilon=0.2, attempts=10,
                   normalize=False, n_channels=3, plot=False):
    """
    Applies K-Means on an image.
    Args:
        image: image
        k: number of clusters required at end
        max_iter: max number of iterations after which K-Means will stop
        epsilon: required accuracy
        attempts:  Flag to specify the number of times the algorithm is
         executed using different initial labellings. The algorithm
         returns the labels that yield the best compactness.
         This compactness is returned as output. See also:
         https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
        normalize: True if the image does not have values between 0 and 255
        n_channels: number of dimensions of the vector on which to apply
         K-Means
        plot: True to show the segmented image

    Returns:
        The image segmented with K-Means.
    """
    flag = cv.KMEANS_PP_CENTERS
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iter,
                epsilon)
    if n_channels == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if plot:
        # show the image
        plt.imshow(image)
        plt.show()

    if normalize:
        # normalize img to 0-255
        norm_image = cv.normalize(image, None, alpha=0, beta=255,
                                  norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        image = norm_image.astype(np.uint8)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, n_channels))
    # convert to float
    pixel_values = np.float32(pixel_values)

    k = k
    # some documentation:
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    attempts = attempts
    compactness, labels, _ = cv.kmeans(
        pixel_values, k, None, criteria, attempts, flag)

    # convert back to 8 bit values
    centers = get_colors_of_clusters(n_clusters=k)
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(
        image.shape[0], image.shape[1], 3)
    if plot:
        # show the image
        plt.imshow(segmented_image)
        plt.show()

    return segmented_image, compactness


def k_means_img_patch_rgb_raw(img, patch_size, k, max_iter, epsilon, attempts,
                              normalize, weight_original_img=0.4,
                              weight_colored_patch=0.4, gamma=0,
                              n_channels=3,
                              compute_also_nn_interpolation=True):
    """
    Steps:
     - Applies K-Means on a patch of img and upscale the result to the shape
       of img with bilinear interpolation
     - It then overlays the upscaled result of K-Means to the original image
       img
     - It can also do the first two steps using nearest neighbor interpolation
       if compute_also_nn_interpolation is True
    Args:
        img: original image
        patch_size: size of the patch of the image
        k: number of clusters required at end
        max_iter: max number of iterations after which K-Means will stop
        epsilon: required accuracy
        attempts:  Flag to specify the number of times the algorithm is
         executed using different initial labellings. The algorithm
         returns the labels that yield the best compactness.
         This compactness is returned as output. See also:
         https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
        normalize: True if the image does not have values between 0 and 255
        weight_original_img: see
         https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
        weight_colored_patch: see
         https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
        gamma: see
         https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
        n_channels = number of dimensions of the vector on which to apply
         K-Means
        compute_also_nn_interpolation: True to apply the described steps
         also using nearest neighbor interpolation

    Returns:
        The segmented patch of image img with K-Means upscaled with bilinear
        interpolation, and optionally also the segmented patch of image img
        with K-Means upscaled with nearest neighbor interpolation
    """
    full_size = img.shape[0]

    patch = downsample_img(img, patch_size, patch_size, False)
    seg_patch, _ = k_means_on_img(
        patch, k=k, max_iter=max_iter, epsilon=epsilon, attempts=attempts,
        normalize=normalize, n_channels=n_channels, plot=False)

    seg_full_bilinear_interp = downsample_img(
        seg_patch, full_size, full_size, False,
        interpolation_method=cv.INTER_LINEAR)

    # Convert the original image as grayscale image to put it in the background
    # to be able to put the patch (the output of Kmeans) over it in a
    # transparent way.
    # Single channel grayscale image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Repeat the grayscale image along all the 3 channels
    stacked_img = np.stack((img_gray,) * 3, axis=-1)

    final_img_bilinear_interp = cv.addWeighted(
        stacked_img, weight_original_img, seg_full_bilinear_interp,
        weight_colored_patch, gamma)

    final_img_nn_interp = None
    if compute_also_nn_interpolation:
        seg_full_nn_interp = downsample_img(seg_patch, full_size, full_size, False,
                                            interpolation_method=cv.INTER_NEAREST)
        final_img_nn_interp = cv.addWeighted(
            stacked_img, weight_original_img, seg_full_nn_interp,
            weight_colored_patch, gamma)

    return final_img_bilinear_interp, final_img_nn_interp
