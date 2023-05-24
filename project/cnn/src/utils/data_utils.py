import numpy as np
import random

from osgeo import gdal


def imread(file_path):
    img = gdal.Open(file_path)
    bands = img.RasterCount
    rows, cols = img.RasterYSize, img.RasterXSize

    image = np.zeros((rows, cols, bands))

    for band_index in range(bands):
        bandx = img.GetRasterBand(band_index + 1)
        datax = bandx.ReadAsArray()
        image[:, :, band_index] = datax

    return image


def normalize_images(images):
    normalized_images = images / 2000
    return normalized_images


def create_crops(image, crop_height, crop_width, stride, num_bands):
    cropped_images = []

    for y in range(0, image.shape[0] - crop_height + 1, stride):
        for x in range(0, image.shape[1] - crop_width + 1, stride):
            cropped_image = image[y : y + crop_height, x : x + crop_width, :num_bands]
            cropped_images.append(cropped_image)

    return np.array(cropped_images)


def choice_im(image_shape, crop_height, crop_width, stride):
    x = random.choice([xx for xx in range(0, image_shape[1] - crop_width + 1, stride)])
    y = random.choice([xx for xx in range(0, image_shape[0] - crop_height + 1, stride)])
    return x, y
