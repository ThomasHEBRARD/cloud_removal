import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from src.dataset.make_dataset import DataLoader

from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Input, Conv2D, Dropout, UpSampling2D, concatenate

BANDS = ["B04", "B03", "B02", "B08"]

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


def build_generator(input_channel):
    """U-Net Generator"""
    gf = 64

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
    d0 = Input(shape=(256, 256, input_channel))

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)
    d5 = conv2d(d4, gf * 8)
    d6 = conv2d(d5, gf * 8)
    d7 = conv2d(d6, gf * 8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf * 8)
    u2 = deconv2d(u1, d5, gf * 8)
    u3 = deconv2d(u2, d4, gf * 8)
    u4 = deconv2d(u3, d3, gf * 4)
    u5 = deconv2d(u4, d2, gf * 2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(
        3,
        kernel_size=4,
        strides=1,
        padding="same",
        activation="tanh",
    )(u7)

    return Model(d0, output_img)

# Calculate 5% of total epochs
total_epochs = 100
save_every_epochs = total_epochs // 20  # Integer division

##############################################################################################
##############################################################################################
##############################################################################################

model = build_generator(input_channel=6)
# Compile the model
model.compile(optimizer=Adam(config["lr"], 0.5), loss=config["loss"], metrics=['accuracy'])

checkpoint = ModelCheckpoint(f'models/{RUN_NAME}/'+'model-{epoch:03d}.h5', period=save_every_epochs)

class ImageCallback(Callback):
    def __init__(self, input_images, ground_truth_images, output_dir):
        super().__init__()
        self.input_images = input_images
        self.ground_truth_images = ground_truth_images
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.input_images)

        fig, axs = plt.subplots(10, 3, figsize=(15, 50))

        for i, (input_image, ground_truth_image, prediction) in enumerate(zip(self.input_images, self.ground_truth_images, predictions)):
            if i > 9:
                break
            prediction_image = ((prediction + 1) / 2 * 255).astype(np.uint8)
            input_image = ((input_image[:,:,2:5] + 1) / 2 * 255).astype(np.uint8)
            ground_truth_image = (ground_truth_image*255).astype(np.uint8)

            axs[i, 0].imshow(input_image)
            axs[i, 0].set_title('Input')
            axs[i, 0].axis('off')

            axs[i, 2].imshow(ground_truth_image)
            axs[i, 2].set_title('Ground Truth')
            axs[i, 2].axis('off')

            axs[i, 1].imshow(prediction_image)
            axs[i, 1].set_title('Prediction')
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/images_at_epoch_{epoch+1}.png')
        plt.close(fig)
        
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []


    def on_epoch_end(self, epoch, logs={}):
        with open(f"models/{RUN_NAME}/loss_log.txt", "a") as f:
            f.write('{},{},{},{}\n'.format(epoch, logs.get('loss'), logs.get('val_loss'), logs.get('accuracy'), logs.get('val_accuracy')))

# Initialize the callback
history = LossHistory()

data_loader = DataLoader()
imgs_A, imgs_B = zip( 
    *data_loader.load_batch(
        bands=BANDS, batch_size=config["batch_size"]
    )
)

imgs_B, imgs_A = np.array(imgs_B), np.array(imgs_A)

callback = ImageCallback(imgs_B, imgs_A, f"models/{RUN_NAME}")

val_A, val_B = imgs_A[:10,:,: ,:], imgs_B[:10,:,:,:]

model.fit(imgs_B, imgs_A, steps_per_epoch=config["nb_batches_per_epoch"], epochs=total_epochs, callbacks=[checkpoint, callback, history], validation_data=(val_B, val_A))
