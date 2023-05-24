import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from src.dataset.make_dataset import DataLoader

from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

BANDS = ["B04", "B03", "B02", "B08"]

def build_unet(input_shape=(256, 256, 6), output_channels=3):
    inputs = Input(input_shape)

    # Contracting Path
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Expanding Path
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4, up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(output_channels, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    return model

# Calculate 5% of total epochs
total_epochs = 100
save_every_epochs = total_epochs // 20  # Integer division

##############################################################################################
##############################################################################################
##############################################################################################

model = build_unet(input_shape=(256, 256, 6), output_channels=3)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

checkpoint = ModelCheckpoint('m/model-{epoch:03d}.h5', period=save_every_epochs)
class ImageCallback(Callback):
    def __init__(self, input_image, output_dir):
        super().__init__()
        self.input_image = input_image
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.input_image)
        plt.imsave(f'{self.output_dir}/image_at_epoch_{epoch+1}.png', ((prediction[0] + 1) / 2 * 255).astype(np.uint8))

data_loader = DataLoader()
imgs_A, imgs_B = zip( 
    *data_loader.load_batch(
        bands=["B04", "B03", "B02", "B08"], batch_size=200
    )
)

imgs_A, imgs_B = np.array(imgs_A), np.array(imgs_B)
callback = ImageCallback(imgs_B[:10, :, :, :], 'output_images')

val_A, val_B = imgs_A[:10,:,:,:], imgs_B[:10,:,:,:]

model.fit(imgs_B, imgs_A, epochs=total_epochs, batch_size=10, callbacks=[checkpoint, callback], validation_data=(val_B, val_A))
