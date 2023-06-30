import os
import json
import numpy as np

from osgeo import gdal

class DataLoader:
    def __init__(self, path=""):
        self.path = path
        with open(
            "/Users/thomashebrard/thesis/code/preprocess/data/dataset_train.json",
            "r",
        ) as f:
            self.dataset_train = json.load(f)
        with open(
            "/Users/thomashebrard/thesis/code/preprocess/data/dataset_test.json",
            "r",
        ) as f:
            self.dataset_test = json.load(f)
        with open(
            "/Users/thomashebrard/thesis/code/preprocess/data/dataset_validation.json",
            "r",
        ) as f:
            self.dataset_validation = json.load(f)

        self.dataset_train_keys = list(self.dataset_train.keys())

    def load_batch(self, bands=["B04", "B03", "B02", "B08"], batch_size=1, is_testing=False, take_val_data=False, val=False):
        batched_data = []
        dataset = {}
        if is_testing:
            dataset = self.dataset_test
            keys = list(dataset.keys())
        elif take_val_data:
            dataset = self.dataset_validation
            keys = list(dataset.keys())
        else:
            dataset = self.dataset_train
            keys = self.dataset_train_keys

        np.random.shuffle(keys)

        data_keys = keys[batch_size:2*batch_size]

        self.n_batches = int(len(list(keys)) / batch_size)

        os.chdir("/Users/thomashebrard/thesis/code/preprocess/")
        
        if self.path:
            date = self.path.split("_")[2]
            key = None
            for k, v in self.dataset_validation.items():
                if date in v["s2_cloudy_B02"]:
                    key = k
            if key:
                data_keys = [str(key)]
            else:
                print("no data")

        for k in data_keys:
            s1_hv = {"im": dataset[k]["s1_hv"], "data": gdal.Open(dataset[k]["s1_hv"]).ReadAsArray()}
            s1_vv = {"im": dataset[k]["s1_vv"], "data": gdal.Open(dataset[k]["s1_vv"]).ReadAsArray()}

            s2_cloudy = {
                band: {"im": dataset[k][f"s2_cloudy_{band}"], "data": gdal.Open(dataset[k][f"s2_cloudy_{band}"]).ReadAsArray()}
                for band in bands
            }
            s2_cloudfree = {
                band: {"im": dataset[k][f"s2_cloud_free_{band}"], "data": gdal.Open(dataset[k][f"s2_cloud_free_{band}"]).ReadAsArray()}
                for band in ["B04", "B03", "B02", "B08"]
            }

            ##### CLIP DATA #####
            CLIP_MIN_S1, CLIP_MAX_S1 = -28000, 13000

            s1_hv["data"] = np.clip(s1_hv["data"], CLIP_MIN_S1, CLIP_MAX_S1)
            s1_vv["data"] = np.clip(s1_vv["data"], CLIP_MIN_S1, CLIP_MAX_S1)

            s1_hv["data"] = ((s1_hv["data"] - CLIP_MIN_S1) / (CLIP_MAX_S1 - CLIP_MIN_S1) * 2) - 1
            s1_vv["data"] = ((s1_vv["data"] - CLIP_MIN_S1) / (CLIP_MAX_S1 - CLIP_MIN_S1) * 2) - 1

            CLIP_MIN_S2, CLIP_MAX_S2 = 0, 3500
            for band in bands:
                s2_cloudy[band]["data"] = np.clip(s2_cloudy[band]["data"], CLIP_MIN_S2, CLIP_MAX_S2)
                s2_cloudy[band]["data"] = (s2_cloudy[band]["data"] - CLIP_MIN_S2) / (
                    CLIP_MAX_S2 - CLIP_MIN_S2
                ) * 2 - 1

            for band in ["B04", "B03", "B02", "B08"]:
                s2_cloudfree[band]["data"] = np.clip(
                    s2_cloudfree[band]["data"], CLIP_MIN_S2, CLIP_MAX_S2
                )
                s2_cloudfree[band]["data"] = (s2_cloudfree[band]["data"] - CLIP_MIN_S2) / (
                    CLIP_MAX_S2 - CLIP_MIN_S2
                ) * 2 - 1

            extra_cloudy_input = {
                k: v for k, v in s2_cloudy.items() if k not in ["B04", "B03", "B02"]
            }

            input = np.stack(
                (
                    s1_hv,
                    s1_vv,
                    s2_cloudy["B04"],
                    s2_cloudy["B03"],
                    s2_cloudy["B02"],
                    *extra_cloudy_input.values(),
                ),
                axis=-1,
            )
            ground_truth = np.stack(
                (s2_cloudfree["B04"], s2_cloudfree["B03"], s2_cloudfree["B02"], s2_cloudfree["B08"]),
                axis=-1,
            )

            if np.mean(s2_cloudfree["B04"]["data"]) > 2000:
                continue

            batched_data.append((ground_truth, input))

        os.chdir("/Users/thomashebrard/thesis/code/project/pix2pix/")

        del s1_hv
        del s1_vv

        del s2_cloudy
        del s2_cloudfree

        del extra_cloudy_input

        del input
        del ground_truth

        if not len(batched_data) == batch_size:
            return self.load_batch(batch_size=batch_size, is_testing=is_testing)
        
        if val:
            val_batched = self.load_batch(bands=bands, batch_size=batch_size, is_testing=is_testing, take_val_data=True)
            return batched_data, val_batched

        return batched_data
