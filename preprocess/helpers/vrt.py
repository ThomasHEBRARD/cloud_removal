import time
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt


def tif_to_vrt(images_path: list, vrt_output: str):
    gdal.BuildVRT(vrt_output, images_path)
    print("VRT build done.")


def display_vrt(vrt_path, show=False):
    dataset = gdal.Open(vrt_path)

    band = dataset.GetRasterBand(1)
    image_data = band.ReadAsArray()

    if show:
        plt.imshow(image_data, cmap="gray")
        plt.title("VRT Image")
        plt.show()

    return image_data


def from_bands_to_vrt(bands, name, rgb_vrt_path, translate_vrt_path, show=False):
    try:
        red_ds = gdal.Open(bands["B04"])
        green_ds = gdal.Open(bands["B03"])
        blue_ds = gdal.Open(bands["B02"])

        vrt_options = gdal.BuildVRTOptions(resolution="highest", separate=True)
        gdal.BuildVRT(
            f"{rgb_vrt_path}/{name}.vrt",
            [red_ds, green_ds, blue_ds],
            options=vrt_options,
        )

        time.sleep(5)

        translate_options = gdal.TranslateOptions(unscale=True)
        gdal.Translate(
            srcDS=f"{rgb_vrt_path}/{name}.vrt",
            destName=f"{translate_vrt_path}/{name}.vrt",
            options=translate_options,
        )

        time.sleep(1)

        translate_ds = gdal.Open(f"{translate_vrt_path}/{name}.vrt")
        im_data = translate_ds.ReadAsArray()

        if show:
            plt.imshow(np.transpose(im_data / 2000, (1, 2, 0)))
            plt.show()
    except:
        print("fail", name, bands)


def show_vrt_image(path):
    translate_ds = gdal.Open(path)
    im_data = translate_ds.ReadAsArray()

    plt.imshow(np.transpose(im_data / 2000, (1, 2, 0)))
    plt.show()
