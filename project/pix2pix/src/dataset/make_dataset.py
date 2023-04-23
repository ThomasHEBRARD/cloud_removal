import os
import json
import numpy as np

from osgeo import gdal

class DataLoader:
    def __init__(self):
        with open(
            "/Users/thomashebrard/thesis/code/preprocess/data/dataset_filtered_water_train.json",
            "r",
        ) as f:
            self.dataset_train = json.load(f)
        with open(
            "/Users/thomashebrard/thesis/code/preprocess/data/dataset_filtered_water_test.json",
            "r",
        ) as f:
            self.dataset_test = json.load(f)

        self.dataset_train_keys = list(self.dataset_train.keys())[:10000]

    def load_batch(self, bands=["B04", "B03", "B02"], batch_size=1, is_testing=False):
        batched_data = []
        dataset = {}
        if is_testing:
            dataset = self.dataset_test
            keys = list(dataset.keys())
        else:
            dataset = self.dataset_train
            keys = self.dataset_train_keys

        np.random.shuffle(keys)

        data_keys = keys[:batch_size]

        self.n_batches = int(len(list(keys)) / batch_size)

        os.chdir("/Users/thomashebrard/thesis/code/preprocess/")

        for k in data_keys:
            s1_hv = gdal.Open(dataset[k]["s1_hv"]).ReadAsArray()
            s1_vv = gdal.Open(dataset[k]["s1_vv"]).ReadAsArray()

            s2_cloudy = {
                band: gdal.Open(dataset[k][f"s2_cloudy_{band}"]).ReadAsArray()
                for band in bands
            }
            s2_cloudfree = {
                band: gdal.Open(dataset[k][f"s2_cloud_free_{band}"]).ReadAsArray()
                for band in ["B04", "B03", "B02"]
            }

            ##### CLIP DATA
            CLIP_MIN_S1, CLIP_MAX_S1 = -28000, 13000

            s1_hv = np.clip(s1_hv, CLIP_MIN_S1, CLIP_MAX_S1)
            s1_vv = np.clip(s1_vv, CLIP_MIN_S1, CLIP_MAX_S1)

            scaled_s1_hv = ((s1_hv - CLIP_MIN_S1) / (CLIP_MAX_S1 - CLIP_MIN_S1) * 2) - 1
            scaled_s1_vv = ((s1_vv - CLIP_MIN_S1) / (CLIP_MAX_S1 - CLIP_MIN_S1) * 2) - 1

            CLIP_MIN_S2, CLIP_MAX_S2 = 100, 5800
            for band in bands:
                s2_cloudy[band] = np.clip(s2_cloudy[band], CLIP_MIN_S2, CLIP_MAX_S2)
                s2_cloudy[band] = (s2_cloudy[band] - CLIP_MIN_S2) / (
                    CLIP_MAX_S2 - CLIP_MIN_S2
                ) * 2 - 1

            for band in ["B04", "B03", "B02"]:
                s2_cloudfree[band] = np.clip(
                    s2_cloudfree[band], CLIP_MIN_S2, CLIP_MAX_S2
                )
                s2_cloudfree[band] = (s2_cloudfree[band] - CLIP_MIN_S2) / (
                    CLIP_MAX_S2 - CLIP_MIN_S2
                ) * 2 - 1

            extra_cloudy_input = {
                k: v for k, v in s2_cloudy.items() if k not in ["B04", "B03", "B02"]
            }

            input = np.stack(
                (
                    scaled_s1_hv,
                    scaled_s1_vv,
                    s2_cloudy["B04"],
                    s2_cloudy["B03"],
                    s2_cloudy["B02"],
                    *extra_cloudy_input.values(),
                ),
                axis=-1,
            )
            ground_truth = np.stack(
                (s2_cloudfree["B04"], s2_cloudfree["B03"], s2_cloudfree["B02"]),
                axis=-1,
            )

            if np.mean(s2_cloudfree["B04"]) > 2000:
                continue

            batched_data.append((ground_truth, input))

        os.chdir("/Users/thomashebrard/thesis/code/project/pix2pix/")

        del s1_hv
        del s1_vv
        del scaled_s1_hv
        del scaled_s1_vv

        del s2_cloudy
        del s2_cloudfree

        del extra_cloudy_input

        del input
        del ground_truth

        if not len(batched_data) == batch_size:
            return self.load_batch(batch_size=batch_size, is_testing=is_testing)

        return batched_data
