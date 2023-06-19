import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import sys

sys.path.append(".")
sys.path.append("..")

from src.dataset.make_dataset import DataLoader

with open("config.json", "r") as file:
    config = json.load(file)

RUN_NAME = ','.join(f'{k}={v}' for k, v in config.items())

OVERRIDE = True

if not os.path.exists(f"models/{RUN_NAME}"):
    os.makedirs(f"models/{RUN_NAME}")
else:
    if not OVERRIDE:
        print("RUN ALREADY DONE")
        sys.exit()

class Pix2Pix:
    def __init__(
        self, bands=["B02", "B03", "B04", "B08"], lr=0.0002, gf=64, df=64, train=False
    ):
        
        # Input shape
        self.bands = bands
        self.lr = lr
        self.gf = gf
        self.df = df
        self.img_rows = 256
        self.img_cols = 256
        # self.input_channels_generator = len(bands) + 6
        self.input_channels_generator = len(bands) + 2
        self.input_channels_discriminator = 3
        self.output_channels = 3

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        optimizer = Adam(self.lr, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss="mae", optimizer=optimizer, metrics=["accuracy"]
        )

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(
            shape=(self.img_rows, self.img_cols, self.input_channels_discriminator)
        )
        img_B = Input(
            shape=(self.img_rows, self.img_cols, self.input_channels_generator)
        )

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        
        self.combined.compile(
            loss=["mse", "mae"], loss_weights=[1, 100], optimizer=optimizer
        )

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding="same")(
                layer_input
            )
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(
                filters,
                kernel_size=f_size,
                strides=1,
                padding="same",
                activation="relu",
            )(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=(self.img_rows, self.img_cols, self.input_channels_generator))

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        d6 = conv2d(d5, self.gf * 8)
        d7 = conv2d(d6, self.gf * 8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf * 8)
        u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(
            self.output_channels,
            kernel_size=4,
            strides=1,
            padding="same",
            activation="tanh",
        )(u7)

        return Model(d0, output_img)

    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding="same")(
                layer_input
            )
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(
            shape=(self.img_rows, self.img_cols, self.input_channels_discriminator)
        )
        img_B = Input(
            shape=(self.img_rows, self.img_cols, self.input_channels_generator)
        )

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding="same")(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, nb_batches_per_epoch=100, batch_size=1, model_path=None):
        self.epochs = epochs
        start_time = datetime.datetime.now()
        if model_path:
            from keras.models import load_model
            self.generator = load_model(model_path)
            self.generator.compile(loss=["mse", "mae"], loss_weights=[1, 100], optimizer=Adam(self.lr, 0.5)) 

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            self.data_loader = DataLoader()
            for batch_i in range(1, nb_batches_per_epoch + 1):
                imgs_A, imgs_B = zip(
                    *self.data_loader.load_batch(
                        bands=self.bands, batch_size=batch_size
                    )
                )

                imgs_B = np.array([[d["data"] for d in inner_array] for inner_array in np.array(imgs_B)])
                imgs_B = np.transpose(imgs_B, (0, 2, 3, 1))
                imgs_A = np.array([[d["data"] for d in inner_array] for inner_array in np.array(imgs_A)])
                imgs_A = np.transpose(imgs_A, (0, 2, 3, 1))
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s"
                    % (
                        epoch,
                        epochs,
                        batch_i,
                        nb_batches_per_epoch,
                        d_loss[0],
                        100 * d_loss[1],
                        g_loss[0],
                        elapsed_time,
                    )
                )

            self.save_model(
                epoch=epoch,
                d_loss=d_loss[0],
                g_loss=g_loss[0],
                accuracy=100 * d_loss[1],
                start_time=start_time,
                save=epoch % (epochs // 20) == 0,
            )

    def save_model(self, epoch, d_loss, g_loss, accuracy, start_time, save=False):
        file_dir = f"models/{RUN_NAME}"
        os.makedirs(file_dir, exist_ok=True)

        with open(f"{file_dir}/d_loss.txt", "a") as f:
            f.write(str(epoch) + "," + str(d_loss) + "\n")
        with open(f"{file_dir}/g_loss.txt", "a") as f:
            f.write(str(epoch) + "," + str(g_loss) + "\n")

        if save:
            self.generator.save(
                f"{file_dir}/model_epoch_{epoch}/model_epoch_{epoch}.h5"
            )
            with open(
                f"{file_dir}/config.json",
                "w",
            ) as f:
                with open(f"config.json", "r") as conf:
                    config = json.load(conf)
                    json.dump(config, f)

        savemode_data_loader = DataLoader(path="S2_32VNH_20210404_B02_17_543500_6297440_256")

        ground_truth, input = zip(
            *savemode_data_loader.load_batch(
                bands=self.bands, batch_size=1, is_testing=True
            )
        )

        input = np.array(input)
        input_pred = np.array([[d["data"] for d in inner_array] for inner_array in np.array(input)])
        input_pred = np.transpose(input_pred, (0, 2, 3, 1))

        # ground_truth = (((np.array(ground_truth) + 1) / 2) * 255).astype(np.uint8)
        ground_truth = np.array([[d["data"] for d in inner_array] for inner_array in np.array(ground_truth)])
        ground_truth = np.transpose(ground_truth, (0, 2, 3, 1))
        ground_truth = ((ground_truth + 1) / 2 * 255).astype(np.uint8)

        output = self.generator.predict(input_pred)

        generated_image = ((output + 1) / 2 * 255).astype(np.uint8)

        input_dict = {
            "s1_hv": {"title": "S1 HV", "desc": input[0][0]["im"].split("/")[-1], "image": input_pred[:, :, :, 0]},
            "s1_vv": {"title": "S1 VV", "desc": input[0][1]["im"].split("/")[-1], "image": input_pred[:, :, :, 1]},
        }

        for i in range(2, len(self.bands) + 2):
            input_dict[f"s2_{self.bands[i - 2]}"] = {
                "title": f"S2 {self.bands[i - 2]}",
                "desc": input[0][i]["im"].split("/")[-1],
                "image": (((input_pred[:, :, :, i] + 1) / 2) * 255).astype(np.uint8),
            }

        #####################################
        ########     PLOT THE INPUT  ########
        #####################################

        for idx_img in range(1):
            num_images = len(self.bands) + 2
            num_rows = (num_images + 3) // 4

            fig1, axes1 = plt.subplots(num_rows, 4, figsize=(16, 8))
            fig1.suptitle("Input")

            for i in range(num_rows):
                for j in range(4):
                    idx = i * 4 + j
                    if idx < num_images:
                        ax = axes1[i, j] if num_rows > 1 else axes1[j]
                        if idx == 0:
                            ax.imshow(input_dict["s1_hv"]["image"][idx_img])
                            ax.set_title("_".join(input_dict["s1_hv"]["desc"].split("_")[:5]))
                            ax.axis("off")
                        elif idx == 1:
                            ax.imshow(input_dict["s1_vv"]["image"][idx_img])
                            ax.set_title("_".join(input_dict["s1_vv"]["desc"].split("_")[:5]))
                            ax.axis("off")
                        else:
                            band = self.bands[idx - 2]
                            ax.imshow(input_dict[f"s2_{band}"]["image"][idx_img])
                            ax.set_title("_".join(input_dict[f"s2_{band}"]["desc"].split("_")[:5]))
                            ax.axis("off")
                    else:
                        (axes1[i, j] if num_rows > 1 else axes1[j]).axis("off")

            if save:
                fig1.savefig(
                    f"models/{RUN_NAME}/model_epoch_{epoch}/input.png"
                )
                with open(f"{file_dir}/model_epoch_{epoch}/input.txt", "a") as f:
                    f.write(input_dict["s1_hv"]["desc"].split(".")[0] + "\n")
                    f.write(input_dict["s1_vv"]["desc"].split(".")[0] + "\n")
                    try:
                        f.write(input_dict[f"s2_B02"]["desc"].split(".")[0] + "\n")
                    except:
                        pass

            else:
                directory = (
                    f"vis/{RUN_NAME}/epoch_{epoch}/"
                )
                if not os.path.exists(directory):
                    os.makedirs(directory)
                fig1.savefig(
                    f"vis/{RUN_NAME}/epoch_{epoch}/input.png"
                )

            #####################################
            ########     PLOT THE OUTPUT ########
            #####################################

            fig2, axes2 = plt.subplots(1, 3, figsize=(16, 8))
            fig2.suptitle(
                f"Epoch : {epoch}/{self.epochs}, lr: {self.lr}, g_loss: {round(g_loss, 2)}, d_loss: {round(d_loss, 2)}, accuracy: {round(accuracy, 2)}%"
            )
            if "s2_B02" in input_dict or "s2_B03" in input_dict or "s2_B04" in input_dict:
                cloudy_input = np.stack(
                    (
                        input_dict[f"s2_B04"]["image"][idx_img],
                        input_dict[f"s2_B03"]["image"][idx_img],
                        input_dict[f"s2_B02"]["image"][idx_img],
                    ),
                    axis=-1,
                )

                axes2[0].imshow(cloudy_input)
                axes2[0].set_title(f"Sentinel-2 Cloudy Input")
                axes2[0].axis("off")

            axes2[1].imshow(generated_image[idx_img])
            axes2[1].set_title(f"Generated Output")
            axes2[1].axis("off")

            axes2[2].imshow(ground_truth[idx_img])
            axes2[2].set_title(f"Ground Truth")
            axes2[2].axis("off")

            plt.tight_layout()

            if save:
                fig2.savefig(
                    f"models/{RUN_NAME}/model_epoch_{epoch}/result.png"
                )
            else:
                fig2.savefig(
                    f"vis/{RUN_NAME}/epoch_{epoch}/result.png"
                )

            plt.close()
