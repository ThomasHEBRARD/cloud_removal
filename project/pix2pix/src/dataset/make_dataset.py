import os
import json
import random
from osgeo import gdal
import numpy as np

from src.utils.data_utils import imread, normalize_images, create_crops, choice_im


class DataLoader:
    def __init__(self):
        with open(
            "/Users/thomashebrard/thesis/code/preprocess/data/dataset_train.json", "r"
        ) as f:
            self.dataset_train = json.load(f)
        with open(
            "/Users/thomashebrard/thesis/code/preprocess/data/dataset_test.json", "r"
        ) as f:
            self.dataset_test = json.load(f)

        self.dataset_train_keys = list(self.dataset_train.keys())[:100]

    def load_batch(self, batch_size=1, is_testing=False):
        batched_data = []
        dataset = {}
        if is_testing:
            dataset = self.dataset_test
            keys = self.dataset_train_keys
        else:
            dataset = self.dataset_train
            keys = list(self.dataset_test.keys())

        np.random.shuffle(keys)
        data_keys = keys[:batch_size]
        self.n_batches = int(len(list(keys)) / batch_size)

        os.chdir("/Users/thomashebrard/thesis/code/preprocess/")

        for k in data_keys:
            s1_hv = gdal.Open(dataset[k]["s1_hv"]).ReadAsArray()
            s1_vv = gdal.Open(dataset[k]["s1_vv"]).ReadAsArray()
            s2_cloudy_B02 = gdal.Open(dataset[k]["s2_cloudy_B02"]).ReadAsArray()
            s2_cloudy_B03 = gdal.Open(dataset[k]["s2_cloudy_B03"]).ReadAsArray()
            s2_cloudy_B04 = gdal.Open(dataset[k]["s2_cloudy_B04"]).ReadAsArray()
            s2_cloudfree_B02 = gdal.Open(dataset[k]["s2_cloud_free_B02"]).ReadAsArray()
            s2_cloudfree_B03 = gdal.Open(dataset[k]["s2_cloud_free_B03"]).ReadAsArray()
            s2_cloudfree_B04 = gdal.Open(dataset[k]["s2_cloud_free_B04"]).ReadAsArray()

            scaled_s1_hv = (
                (s1_hv - np.min(s1_hv)) / (np.max(s1_hv) - np.min(s1_hv)) * 2
            ) - 1
            scaled_s1_vv = (
                (s1_vv - np.min(s1_vv)) / (np.max(s1_vv) - np.min(s1_vv)) * 2
            ) - 1

            scaled_s2_cloudy_B02 = (
                (s2_cloudy_B02 - np.min(s2_cloudy_B02))
                / (np.max(s2_cloudy_B02) - np.min(s2_cloudy_B02))
                * 2
            ) - 1
            scaled_s2_cloudy_B03 = (
                (s2_cloudy_B03 - np.min(s2_cloudy_B03))
                / (np.max(s2_cloudy_B03) - np.min(s2_cloudy_B03))
                * 2
            ) - 1
            scaled_s2_cloudy_B04 = (
                (s2_cloudy_B04 - np.min(s2_cloudy_B04))
                / (np.max(s2_cloudy_B04) - np.min(s2_cloudy_B04))
                * 2
            ) - 1

            scaled_s2_cloudfree_B02 = (
                (s2_cloudfree_B02 - np.min(s2_cloudfree_B02))
                / (np.max(s2_cloudfree_B02) - np.min(s2_cloudfree_B02))
                * 2
            ) - 1
            scaled_s2_cloudfree_B03 = (
                (s2_cloudfree_B03 - np.min(s2_cloudfree_B03))
                / (np.max(s2_cloudfree_B03) - np.min(s2_cloudfree_B03))
                * 2
            ) - 1
            scaled_s2_cloudfree_B04 = (
                (s2_cloudfree_B04 - np.min(s2_cloudfree_B04))
                / (np.max(s2_cloudfree_B04) - np.min(s2_cloudfree_B04))
                * 2
            ) - 1

            input = np.stack(
                (
                    scaled_s1_hv,
                    scaled_s1_vv,
                    scaled_s2_cloudy_B04,
                    scaled_s2_cloudy_B03,
                    scaled_s2_cloudy_B02,
                ),
                axis=-1,
            )
            ground_truth = np.stack(
                (
                    scaled_s2_cloudfree_B04,
                    scaled_s2_cloudfree_B03,
                    scaled_s2_cloudfree_B02,
                ),
                axis=-1,
            )

            batched_data.append((ground_truth, input))

        os.chdir("/Users/thomashebrard/thesis/code/project/pix2pix/")

        del s1_hv
        del s1_vv
        del s2_cloudy_B02
        del s2_cloudy_B03
        del s2_cloudy_B04
        del s2_cloudfree_B02
        del s2_cloudfree_B03
        del s2_cloudfree_B04
        del scaled_s1_hv
        del scaled_s1_vv

        del scaled_s2_cloudy_B02
        del scaled_s2_cloudy_B03
        del scaled_s2_cloudy_B04

        del scaled_s2_cloudfree_B02
        del scaled_s2_cloudfree_B03
        del scaled_s2_cloudfree_B04
        del input
        del ground_truth

        return batched_data
