import numpy as np

from src.utils.data_utils import imread, normalize_images, create_crops


class DataLoader:
    # def load_data(self, batch_size=1, is_testing=False):
    #     data_type = "train" if not is_testing else "test"
    #     path = glob("./datasets/%s/%s/*" % (self.dataset_name, data_type))

    #     batch_images = np.random.choice(path, size=batch_size)

    #     imgs_A = []
    #     imgs_B = []
    #     for img_path in batch_images:
    #         img = self.imread(img_path)

    #         h, w, _ = img.shape
    #         _w = int(w / 2)
    #         img_A, img_B = img[:, :_w, :], img[:, _w:, :]

    #         img_A = scipy.misc.imresize(img_A, self.img_res)
    #         img_B = scipy.misc.imresize(img_B, self.img_res)

    #         # If training => do random flip
    #         if not is_testing and np.random.random() < 0.5:
    #             img_A = np.fliplr(img_A)
    #             img_B = np.fliplr(img_B)

    #         imgs_A.append(img_A)
    #         imgs_B.append(img_B)

    #     imgs_A = np.array(imgs_A) / 127.5 - 1.0
    #     imgs_B = np.array(imgs_B) / 127.5 - 1.0

    #     return imgs_A, imgs_B

    def load_data(self):
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

        return cropped_images_A[:10,:,:,:], cropped_images_B[:10,:,:,:]
