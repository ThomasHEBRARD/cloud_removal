import json
import random
import numpy as np

from src.utils.data_utils import imread, normalize_images, create_crops, choice_im


class DataLoader:
    def generator_data(self):
        cloud_free_vrts = ["data/vrt/s2/translate/S2_32VNH_20200601.vrt"]
        cloudy_vrts = ["data/vrt/s2/translate/S2_32VNH_20200611.vrt"]
        s1_vrts = ["data/vrt/s1/S1_32VNH_20200609_D_139.vrt"]

        cloudy_images = [imread(file_path) for file_path in cloudy_vrts]
        cloud_free_images = [imread(file_path) for file_path in cloud_free_vrts]
        s1 = [imread(file_path) for file_path in s1_vrts]

        cloudy_images = cloudy_images[0]
        cloud_free_images = cloud_free_images[0]
        s1 = s1[0]
        s1 = s1[:, :, :2]

        normalized_cloudy_images = [
            normalize_images(cloudy_image) for cloudy_image in cloudy_images
        ]
        normalized_cloud_free_images = np.array(
            [
                normalize_images(cloud_free_image)
                for cloud_free_image in cloud_free_images
            ]
        )

        input_image = np.concatenate((s1, normalized_cloudy_images), axis=-1)

        x, y = choice_im(input_image.shape, 1024, 1024, 1024)
        cropped_B = np.array([input_image[y : y + 1024, x : x + 1024, :]])
        cropped_A = np.array(
            [normalized_cloud_free_images[y : y + 1024, x : x + 1024, :]]
        )
        print(cropped_A.shape)
        print(cropped_B.shape)
        del input_image
        del normalized_cloud_free_images
        del normalized_cloudy_images
        del cloudy_images
        del s1
        del cloud_free_images

        return cropped_A, cropped_B

    def load_data(
        self,
    ):
        cloud_free_vrts = ["data/vrt/s2/translate/S2_32VNH_20200601.vrt"]
        cloudy_vrts = ["data/vrt/s2/translate/S2_32VNH_20200611.vrt"]
        s1_vrts = ["data/vrt/s1/S1_32VNH_20200609_D_139.vrt"]

        cloudy_images = [imread(file_path) for file_path in cloudy_vrts]
        cloud_free_images = [imread(file_path) for file_path in cloud_free_vrts]
        s1 = [imread(file_path) for file_path in s1_vrts]

        cloudy_images = cloudy_images[0]
        cloud_free_images = cloud_free_images[0]
        s1 = s1[0]
        s1 = s1[:, :, :2]

        normalized_cloudy_images = [
            normalize_images(cloudy_image) for cloudy_image in cloudy_images
        ]
        normalized_cloud_free_images = np.array(
            [
                normalize_images(cloud_free_image)
                for cloud_free_image in cloud_free_images
            ]
        )

        input_image = np.concatenate((s1, normalized_cloudy_images), axis=-1)

        cropped_images_B = create_crops(input_image, 1024, 1024, 1024, 5)
        cropped_images_A = create_crops(
            normalized_cloud_free_images, 1024, 1024, 1024, 3
        )

        del input_image
        del normalized_cloud_free_images
        del normalized_cloudy_images
        del cloudy_images
        del s1
        del cloud_free_images

        return cropped_images_A, cropped_images_B

    def load_batch(self, batch_size):
        with open("dataset.json", "r") as f:
            dataset = json.load(f)
        
        data = random.sample(list(dataset.keys()), batch_size)

        # contruct data, channels etc.
