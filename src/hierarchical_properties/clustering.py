import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering

from src.data_generation.utils import downsample_img, normalize_img


def get_colors_of_clusters(n_clusters: int) -> np.ndarray:
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


def get_segmented_img(feature: np.ndarray, centers: np.ndarray,
                      labels: np.ndarray, plot=False) -> np.ndarray:
    """
    Gets the segmented image of feature, given the centers and labels retrieved
    by a clustering algorithm.
    Args:
        feature: feature on which the clustering method was applied on
        centers: centers of the clusters
        labels: labels of the pixels
        plot: True to plot the segmented image

    Returns:
        The segmented image.
    """
    n_rgb_channels = 3
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to RGB image dimension
    segmented_image = segmented_image.reshape(
        feature.shape[0], feature.shape[1], n_rgb_channels)
    if plot:
        # show the image
        plt.imshow(segmented_image)
        plt.show()

    return segmented_image


def k_means_on_feature(feature: np.ndarray, n_clusters: int, max_iter=100,
                       epsilon=0.2, attempts=10, normalize=False, n_channels=3,
                       plot=False) -> (np.ndarray, float):
    """
    Applies K-Means on a feature extracted by a neural network.
    Args:
        feature: feature extracted by a neural network
        n_clusters: The number of clusters to form as well as the number
         of centroids to generate
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
    if plot:
        # show the image
        plt.imshow(feature)
        plt.show()

    if normalize:
        min_value = 0
        max_value = 255
        feature = normalize_img(feature, min_value, max_value,
                                return_int_values=True)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = feature.reshape((-1, n_channels))
    # convert to float
    pixel_values = np.float32(pixel_values)

    kmeans = KMeans(n_clusters=n_clusters, n_init=attempts,
                    max_iter=max_iter, tol=epsilon).fit(pixel_values)

    # convert back to 8 bit values
    centers = get_colors_of_clusters(n_clusters=n_clusters)

    # get the labels array
    labels = kmeans.labels_

    segmented_image = get_segmented_img(feature, centers, labels, plot=plot)
    # Sum of squared distances of samples to their closest cluster center,
    # weighted by the sample weights if provided.
    compactness = kmeans.inertia_

    return segmented_image, compactness


def ward_on_feature(feature: np.ndarray, n_clusters: np.ndarray,
                    normalize=False, n_channels=3, plot=False) -> np.ndarray:
    """
    Applies Ward's Hierarchical Clustering on a feature extracted by a neural
    network.
    Args:
        feature: feature extracted by a neural network
        n_clusters: The number of clusters to form as well as the number
         of centroids to generate
        normalize: True if the image does not have values between 0 and 255
        n_channels: number of dimensions of the vector on which to apply
         Ward's Hierarchical Clustering
        plot: True to show the segmented image

    Returns:
        The image segmented with Ward's Hierarchical Clustering.
    """
    if plot:
        # show the image
        plt.imshow(feature)
        plt.show()

    if normalize:
        min_value = 0
        max_value = 255
        feature = normalize_img(feature, min_value, max_value,
                                return_int_values=True)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = feature.reshape((-1, n_channels))
    # convert to float
    pixel_values = np.float32(pixel_values)

    agg_ward = AgglomerativeClustering(
        n_clusters=n_clusters, linkage='ward').fit(pixel_values)

    # convert back to 8 bit values
    centers = get_colors_of_clusters(n_clusters=n_clusters)

    # get the labels array
    labels = agg_ward.labels_

    segmented_image = get_segmented_img(feature, centers, labels, plot=plot)
    return segmented_image


def k_means_img_patch_rgb_raw(img: np.ndarray, patch_size: int, k: int,
                              max_iter: int, epsilon: float, attempts: int,
                              normalize: bool, weight_original_img=0.4,
                              weight_colored_patch=0.4, gamma=0,
                              n_channels=3,
                              compute_also_nn_interpolation=True) \
        -> (np.ndarray, np.ndarray):
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
        n_channels: number of dimensions of the vector on which to apply
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
    seg_patch, _ = k_means_on_feature(
        patch, n_clusters=k, max_iter=max_iter, epsilon=epsilon, attempts=attempts,
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
