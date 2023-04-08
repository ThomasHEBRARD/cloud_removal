from osgeo import gdal
import matplotlib.pyplot as plt

# 0 => clear land pixel
# 1 => clear water pixel
# 2 => cloud shadow
# 3 => snow
# 4 => cloud


def display_fmask(tif_path, show=False):
    dataset = gdal.Open(tif_path)

    band = dataset.GetRasterBand(1)
    image_data = band.ReadAsArray()

    image_data[(image_data < 0) | (image_data > 4)] = -9999

    if show:
        plt.imshow(image_data, cmap="gray", vmin=0, vmax=4)
        plt.title("Single-Band TIF Image")
        plt.show()

    return image_data
