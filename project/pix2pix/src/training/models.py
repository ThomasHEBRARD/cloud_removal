import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

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
        self.img_rows = 1024
        self.img_cols = 1024
        self.channels = 5
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        if train:
            self.dataset = DataLoader().load_data()
        else:
            self.dataset = ([], [])

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
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in [0]:
            i = 0
            for imgs_A, imgs_B in zip(self.dataset[0], self.dataset[1]):
                imgs_B = np.array([imgs_B])
                imgs_A = np.array([imgs_A])

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
                        i,
                        i,
                        d_loss[0],
                        100 * d_loss[1],
                        g_loss[0],
                        elapsed_time,
                    )
                )

            if epoch == 0:
                self.save_model(epoch, d_loss[0], g_loss[0], 100 * d_loss[1])
            i += 1
            break

    def save_model(self, epoch, d_loss, g_loss, accuracy):
        self.generator.save(f"models/model_epoch_{epoch}/model_epoch_{epoch}.h5")

        pix = Pix2Pix()
        gen_image = self.generator.predict(pix.dataset[1])

        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        axs[0].imshow(pix.dataset[1][0][:,:,2:])
        axs[0].set_title('Cloudy input')

        axs[1].imshow(gen_image[0])
        axs[1].set_title('Generated output')

        axs[2].imshow(pix.dataset[0][0])
        axs[2].set_title('Ground Truth')

        # Remove axis ticks
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Epoch : {epoch}/200, lr: {self.lr}, g_loss: {g_loss}, d_loss: {d_loss}, accuracy: {accuracy}%")
        fig.savefig(f"models/model_epoch_{epoch}/result.png")
        plt.close()
