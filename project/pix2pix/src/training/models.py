import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal

from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Model
from keras.models import load_model

from keras.optimizers import Adam
import sys

sys.path.append(".")
sys.path.append("..")

from src.dataset.make_dataset import DataLoader


class Pix2Pix:
    def __init__(self, lr=0.0002, gf=64, df=64, train=False):
        # Input shape
        self.lr = lr
        self.gf = gf
        self.df = df
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 5
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        optimizer = Adam(self.lr, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss="mse", optimizer=optimizer, metrics=["accuracy"]
        )

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=(self.img_rows, self.img_cols, 3))
        img_B = Input(shape=(self.img_rows, self.img_cols, 5))

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
        d0 = Input(shape=(self.img_rows, self.img_cols, 5))

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
            3, kernel_size=4, strides=1, padding="same", activation="tanh"
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

        img_A = Input(shape=(self.img_rows, self.img_cols, 3))
        img_B = Input(shape=(self.img_rows, self.img_cols, 5))

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding="same")(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1):
        self.epochs = epochs
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        n_batches_per_epoch = 1000

        for epoch in range(epochs):
            self.data_loader = DataLoader()
            for batch_i in range(1, n_batches_per_epoch + 1):
                imgs_A, imgs_B = zip(
                    *self.data_loader.load_batch(batch_size=batch_size)
                )
                imgs_B = np.array(imgs_B)
                imgs_A = np.array(imgs_A)
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
                        n_batches_per_epoch,
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
        if save:
            self.generator.save(
                f"models/run_{start_time.strftime('%Y-%m-%dT%H:%M:%S')}/model_epoch_{epoch}/model_epoch_{epoch}.h5"
            )
        savemode_data_loader = DataLoader()
        BATCH_SIZE = 3

        fig, axes = plt.subplots(BATCH_SIZE, 5, figsize=(20, 12))

        dA, dB = zip(
            *savemode_data_loader.load_batch(batch_size=BATCH_SIZE, is_testing=True)
        )
        dB = np.array(dB)
        dA = np.array(dA)

        input = dB

        s1_hv = dB[:, :, :, 0]
        s1_vv = dB[:, :, :, 1]

        gen_image = self.generator.predict(input)
        reverted_generated_output = (gen_image + 1) / 2
        reverted_generated_output = (reverted_generated_output * 255).astype(np.uint8)

        s2_cloudy = dB[:, :, :, 2:]
        reverted_s2_cloudy = (s2_cloudy + 1) / 2
        reverted_s2_cloudy = (reverted_s2_cloudy * 255).astype(np.uint8)

        s2_cloud_free = dA
        reverted_s2_cloud_free = (s2_cloud_free + 1) / 2
        reverted_s2_cloud_free = (reverted_s2_cloud_free * 255).astype(np.uint8)

        for i in range(BATCH_SIZE):
            axes[i, 0].imshow(reverted_s2_cloudy[i])
            axes[i, 0].axis("off")
            axes[i, 0].set_title("S2 RGB cloudy input")

            axes[i, 1].imshow(s1_hv[i])
            axes[i, 1].axis("off")
            axes[i, 1].set_title("S1 VH input")

            axes[i, 2].imshow(s1_vv[i])
            axes[i, 2].axis("off")
            axes[i, 2].set_title("S1 VV input")
            # Ploidx output_image in the second block
            axes[i, 3].imshow(reverted_generated_output[i])
            axes[i, 3].axis("off")
            axes[i, 3].set_title("Generated Output")
            # Ploidx truth_image in the third block
            axes[i, 4].imshow(reverted_s2_cloud_free[i])
            axes[i, 4].axis("off")
            axes[i, 4].set_title("Ground Truth")

            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            fig.suptitle(
                f"Epoch : {epoch}/{self.epochs}, lr: {self.lr}, g_loss: {round(g_loss, 2)}, d_loss: {round(d_loss, 2)}, accuracy: {round(accuracy, 2)}%"
            )
            if save:
                fig.savefig(f"models/run_{start_time.strftime('%Y-%m-%dT%H:%M:%S')}/model_epoch_{epoch}/result.png")
            else:
                fig.savefig(f"vis/result_epoch_{epoch}.png")
            plt.close()
