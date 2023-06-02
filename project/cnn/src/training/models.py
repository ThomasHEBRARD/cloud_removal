import sys
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
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

BANDS = ["B04", "B03", "B02", "B08"]

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
lr = 0.0001
# Compile the model
model.compile(optimizer=Adam(lr, 0.5), loss='mse', metrics=['accuracy'])

checkpoint = ModelCheckpoint('m/model-{epoch:03d}.h5', period=save_every_epochs)
class ImageCallback(Callback):
    def __init__(self, input_image, ground_truth_image, output_dir):
        super().__init__()
        self.input_image = input_image
        self.ground_truth_image = ground_truth_image
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.input_image)

        # Select the first image from input, ground truth, and prediction
        input_image = self.input_image[0]
        ground_truth_image = ((self.ground_truth_image[0] + 1) / 2 * 255).astype(np.uint8) 
        
        prediction_image = ((prediction[0] + 1) / 2 * 255).astype(np.uint8)

        # Plot input image
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Input')
        plt.imshow(((input_image[:,:,2:5] + 1) / 2 * 255).astype(np.uint8))
        plt.axis('off')

        # Plot ground truth image
        plt.subplot(1, 3, 2)
        plt.title('Ground Truth')
        plt.imshow(ground_truth_image)
        plt.axis('off')

        # Plot predicted image
        plt.subplot(1, 3, 3)
        plt.title('Prediction')
        plt.imshow(prediction_image)
        plt.axis('off')

        # Save the figure
        plt.savefig(f'{self.output_dir}/image_at_epoch_{epoch+1}.png')
        plt.close()

data_loader = DataLoader()
imgs_A, imgs_B = zip( 
    *data_loader.load_batch(
        bands=["B04", "B03", "B02", "B08"], batch_size=500
    )
)

imgs_A, imgs_B = np.array(imgs_A), np.array(imgs_B)
callback = ImageCallback(imgs_B[:10, :, :, :], imgs_A[:10, :, :, :], 'output_images')

val_A, val_B = imgs_A[:10,:,:,:], imgs_B[:10,:,:,:]

model.fit(imgs_B, imgs_A, epochs=total_epochs, batch_size=10, callbacks=[checkpoint, callback], validation_data=(val_B, val_A))
