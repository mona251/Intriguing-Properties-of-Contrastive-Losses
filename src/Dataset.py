import numpy as np
from src.load_data import load_mnist, sample_uniformly_imgs
from src.utils import crop_digit, downsample_img
from src.create_data import overlay_small_img_on_large_img_at_random_position
from src.create_data import overlay_small_img_on_large_img_grid


class Dataset(object):
    def __init__(self):
        self.mnist_dataset_train, _ = load_mnist()

    def generate_multi_digits_randomly_placed(self, n_images, n_digits_in_img,
                                              img_shape=(112, 112)):
        """
        Generates a dataset with images with multi digits randomly placed on
        black images.
        Args:
            n_images: number of images to generate
            n_digits_in_img: number of digits for each image
            img_shape: shape of each image

        Returns:
            List containing the generated images.
        """
        canvas_height = img_shape[0]
        canvas_width = img_shape[1]

        dataset = []

        for img_idx in range(n_images):
            canvas = np.zeros((canvas_height, canvas_width))
            # Sample the digits to insert in the current image
            sampled_digit_images = sample_uniformly_imgs(
                self.mnist_dataset_train, n_digits_in_img)
            cropped_sampled_digit_images = []
            # Crop the digits
            for sampled_digit_img in sampled_digit_images:
                cropped_sampled_digit_images.append(
                    crop_digit(sampled_digit_img, plot=False))
            # Insert the cropped digits in the current image
            for cropped_sampled_digit_img in cropped_sampled_digit_images:
                overlay_small_img_on_large_img_at_random_position(
                    canvas, cropped_sampled_digit_img,
                    is_large_img_grayscale=True)
            # Add the current image to the list of images
            dataset.append(canvas)

        return dataset

    def generate_multi_digits_grid(self, n_images, n_repetition_digit_img,
                                   n_cells_in_grid=16, img_shape=(112, 112),
                                   repeat_same_digit=False):
        """
        Generates a dataset with images with multi digits placed on a
        grid.
        Args:
            n_images: number of images to generate
            n_repetition_digit_img: number of digits for each image
            n_cells_in_grid: number of cells of the grid
            img_shape: shape of each image
            repeat_same_digit: True to have only the same digit repeated in
             one image

        Returns:
            List containing the generated images.
        """
        canvas_height = img_shape[0]
        canvas_width = img_shape[1]

        dataset = []
        for img_idx in range(n_images):
            sampled_digit_image = sample_uniformly_imgs(
                self.mnist_dataset_train, n_repetition_digit_img)[0]

            canvas = np.zeros((canvas_height, canvas_width))
            if repeat_same_digit:
                canvas = overlay_small_img_on_large_img_grid(
                    sampled_digit_image, canvas, n_cells_in_grid,
                    n_repetition_digit_img, is_large_img_grayscale=True,
                    repeat_small_img=repeat_same_digit,
                    dataset=None,
                    plot=False)
            else:
                canvas = overlay_small_img_on_large_img_grid(
                    sampled_digit_image, canvas, n_cells_in_grid,
                    n_repetition_digit_img, is_large_img_grayscale=True,
                    repeat_small_img=False, dataset=self.mnist_dataset_train,
                    plot=False)
            dataset.append(canvas)

        return dataset

    def generate_two_digits_varying_size_randomly_placed(
            self, n_images, img_shape=(112, 112), min_size=20, max_size=80):
        """
        Generates a dataset with images with two digits randomly placed on
        black images.
        Args:
            n_images: number of images to generate
            img_shape: shape of each image
            min_size: minimum size of the digit
            max_size: maximum size of the digit

        Returns:
            List containing the generated images.
        """
        n_digits = 2
        canvas_height = img_shape[0]
        canvas_width = img_shape[1]

        dataset = []

        for img_idx in range(n_images):
            canvas = np.zeros((canvas_height, canvas_width))

            sampled_digit_images = sample_uniformly_imgs(
                self.mnist_dataset_train, n_digits)

            first_digit = sampled_digit_images[0]
            second_digit = sampled_digit_images[1]

            min_size = min_size
            max_size = max_size
            scale = 10
            size_first_digit = min_size
            size_second_digit = np.random.randint(
                min_size / scale, (max_size / scale) + 1) * scale

            first_digit = downsample_img(first_digit, size_first_digit,
                                         size_first_digit, grayscale=True)
            second_digit = downsample_img(second_digit, size_second_digit,
                                          size_second_digit, grayscale=True)
            first_digit_cropped = crop_digit(first_digit, plot=False)
            second_digit_cropped = crop_digit(second_digit, plot=False)

            overlay_small_img_on_large_img_at_random_position(
                canvas, second_digit_cropped, is_large_img_grayscale=True)
            overlay_small_img_on_large_img_at_random_position(
                canvas, first_digit_cropped, is_large_img_grayscale=True)

            dataset.append(canvas)

        return dataset

    def generate_multi_digits_on_another_dataset_grid(
            self, other_dataset, n_images, n_repetition_digit_img,
            n_cells_in_grid=16):
        sampled_digit_images = sample_uniformly_imgs(
            self.mnist_dataset_train, n_images)

        sampled_other_dataset_images = \
            sample_uniformly_imgs(other_dataset, n_images)

        dataset = []

        for i in range(n_images):
            # TODO: what about the dim of the mnist img? Is it fine to just
            #  downsample/upsample it to the dimension of a cell of a grid?
            sampled_digit_img = sampled_digit_images[i]
            sampled_other_dataset_img = sampled_other_dataset_images[i]

            sampled_other_dataset_img = \
                overlay_small_img_on_large_img_grid(
                    sampled_digit_img, sampled_other_dataset_img,
                    n_cells_in_grid, n_repetition_digit_img,
                    is_large_img_grayscale=False, plot=False)
            dataset.append(sampled_other_dataset_img)
        return dataset
