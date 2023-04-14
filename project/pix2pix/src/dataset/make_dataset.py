import json
import random
from osgeo import gdal
import numpy as np

from src.utils.data_utils import imread, normalize_images, create_crops, choice_im


class DataLoader:
    def __init__(self):
        with open("dataset_train.json", "r") as f:
            self.dataset_train = json.load(f)
        with open("dataset_test.json", "r") as f:
            self.dataset_test = json.load(f)
            
        self.dataset_train_keys = self.dataset_train.keys()

    def load_batch(self, batch_size):
        batched_data = []
        data_keys = random.sample(self.dataset_train_keys, batch_size)
        self.n_batches = int(len(data_keys)/batch_size)

        for k in data_keys:
            s1_hv = gdal.Open(self.dataset_train[k]["s1_hv"]).ReadAsArray()
            s1_vv = gdal.Open(self.dataset_train[k]["s1_vv"]).ReadAsArray()
            s2_cloudy_B02 = gdal.Open(self.dataset_train[k]["s2_cloudy_B02"]).ReadAsArray()
            s2_cloudy_B03 = gdal.Open(self.dataset_train[k]["s2_cloudy_B03"]).ReadAsArray()
            s2_cloudy_B04 = gdal.Open(self.dataset_train[k]["s2_cloudy_B04"]).ReadAsArray()
            s2_cloudfree_B02 = gdal.Open(self.dataset_train[k]["s2_cloudfree_B02"]).ReadAsArray()
            s2_cloudfree_B03 = gdal.Open(self.dataset_train[k]["s2_cloudfree_B03"]).ReadAsArray()
            s2_cloudfree_B04 = gdal.Open(self.dataset_train[k]["s2_cloudfree_B04"]).ReadAsArray()

            input = np.stack((s1_hv, s1_vv, s2_cloudy_B04, s2_cloudy_B03, s2_cloudy_B02), axis=-1)
            ground_truth = np.stack((s2_cloudfree_B04, s2_cloudfree_B03, s2_cloudfree_B02), axis=-1)
            batched_data.append((ground_truth, input))

        del s1_hv
        del s1_vv
        del s2_cloudy_B02
        del s2_cloudy_B03
        del s2_cloudy_B04
        del s2_cloudfree_B02
        del s2_cloudfree_B03
        del s2_cloudfree_B04
        del input
        del ground_truth

        return batched_data