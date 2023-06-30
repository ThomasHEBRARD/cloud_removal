import sys
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")
import tensorflow as tf

from src.training.models import Pix2Pix
from src.dataset.make_dataset import DataLoader

# epoch = 190
# start = "epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path="
# model = load_model(f"/Users/thomashebrard/thesis/code/project/cnn/m/model-100.h5")

models_bands = [
    # (f"/Users/thomashebrard/thesis/code/project/cnn/models/continue_epochs=500,lr=0.0001,batch_size=16,nb_batches_per_epoch=1000,model_path=,loss=mse/model-100.h5", ["B04", "B03", "B02", "B08"], "unet"),
    # (f"/Users/thomashebrard/thesis/code/project/pix2pix/models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy/model_epoch_175/model_epoch_175.h5,v=cloudy/model_epoch_250/model_epoch_250.h5", ["B04", "B03", "B02", "B08"], "gan"),
    # (f"/Users/thomashebrard/thesis/code/project/cnn/models/epochs=500,lr=0.0001,batch_size=16,nb_batches_per_epoch=1000,model_path=,loss=mse/model-010.h5", ["B04", "B03", "B02", "B08"], "unet"),
    # (f"models/run_2023-06-01T15:26:55/model_epoch_60/model_epoch_60.h5", ["B04", "B03", "B02", "B08"], "gan"),
    # (f"models/DO_NOT_DELETE/model_epoch_{190}/model_epoch_{190}.h5", ["B04", "B03", "B02"], "first_gan"),
    (f"models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=/model_epoch_{475}/model_epoch_{475}.h5", ["B04", "B03", "B02", "B08"], "gan"),
    # (f"models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=/model_epoch_{275}/model_epoch_{275}.h5", ["B04", "B03", "B02", "B08"], "gan"),
]

# models_bands = [
#     # (f"/Users/thomashebrard/thesis/code/project/cnn/models/continue_epochs=500,lr=0.0001,batch_size=16,nb_batches_per_epoch=1000,model_path=,loss=mse/model-100.h5", ["B04", "B03", "B02", "B08"], "unet"),
#     (f"models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=/model_epoch_{475}/model_epoch_{475}.h5", ["B04", "B03", "B02", "B08"], "gan"),
#     (f"/Users/thomashebrard/thesis/code/project/pix2pix/models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=,v=cloudy/model_epoch_175/model_epoch_175.h5,v=cloudy/model_epoch_250/model_epoch_250.h5", ["B04", "B03", "B02", "B08"], "gan"),
#     # (f"models/DO_NOT_DELETE/model_epoch_{190}/model_epoch_{190}.h5", ["B04", "B03", "B02"], "first_gan")
# ]


# ################################
BATCH_SIZE = 3
savemode_data_loader = DataLoader()
ground_truth, input = zip(
    *savemode_data_loader.load_batch(
        bands=["B04", "B03", "B02", "B08"], batch_size=BATCH_SIZE, is_testing=True
    )
)

input = np.array(input)
input_pred = np.array([[d["data"] for d in inner_array] for inner_array in np.array(input)])
input_pred = np.transpose(input_pred, (0, 2, 3, 1))

# ground_truth = (((np.array(ground_truth) + 1) / 2) * 255).astype(np.uint8)
ground_truth = np.array([[d["data"] for d in inner_array] for inner_array in np.array(ground_truth)])
ground_truth = np.transpose(ground_truth, (0, 2, 3, 1))
ground_truth = ((ground_truth + 1) / 2 * 255).astype(np.uint8)

input_dict = {
    "s1_hv": {"title": "S1 HV", "desc": input[0][0]["im"].split("/")[-1], "image": input_pred[:, :, :, 0]},
    "s1_vv": {"title": "S1 VV", "desc": input[0][1]["im"].split("/")[-1], "image": input_pred[:, :, :, 1]},
}

for i in range(2, len(["B04", "B03", "B02", "B08"]) + 2):
    input_dict[f"s2_{['B04', 'B03', 'B02', 'B08'][i - 2]}"] = {
        "title": f"S2 {['B04', 'B03', 'B02', 'B08'][i - 2]}",
        "desc": input[0][i]["im"].split("/")[-1],
        "image": (((input_pred[:, :, :, i] + 1) / 2) * 255).astype(np.uint8),
    }

# ########################################################
# ########                  PLOT                  ########
# ########################################################
rows = BATCH_SIZE
cols = 3 + len(models_bands)  # For cloudy_input, ground_truth and each model output
DO = True
# from PIL import Image

# Define the image dimensions and separator size
image_width, image_height = 256, 256  # Replace with your actual image size
separator_size = 10  # Size of white separator between images in pixels
if DO:
    # Lists to store PIL Image objects
    all_images = []

    for image_idx in range(BATCH_SIZE):
        row_images = []

        # Create the cloudy input
        cloudy_input = np.stack(
            (
                input_dict[f"s2_B04"]["image"][image_idx],
                input_dict[f"s2_B03"]["image"][image_idx],
                input_dict[f"s2_B02"]["image"][image_idx],
            ),
            axis=-1,
        )

        def normalize_image(image):
            return (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

        s1 = normalize_image(input_dict["s1_hv"]["image"][image_idx])

        # Convert to PIL Image and resize to match the other images
        gray_image_pil = Image.fromarray(s1.astype(np.uint8)).resize((image_width, image_height), Image.ANTIALIAS)

        # Convert to RGB
        gray_image_rgb = gray_image_pil.convert("RGB")

        row_images.append(Image.fromarray((cloudy_input).astype(np.uint8)))
        row_images.append(Image.fromarray(s1).convert("RGB"))

        # Add the ground truth
        row_images.append(Image.fromarray((ground_truth[image_idx]).astype(np.uint8)))

        # Apply each model and add its output
        for j, (model, bands, filename) in enumerate(models_bands):
            print(f"MODEL: {model}", len(bands)+2, input_pred.shape)
            model = load_model(model)
            
            input_to_show = input_pred[:,:,:,:len(bands)+2]

            output = model.predict(input_to_show)
            generated_image = ((output + 1) / 2 * 400).astype(np.uint8)

            row_images.append(Image.fromarray(generated_image[image_idx]))

        # Stitch together the images for this row
        stitched_row_image = Image.new('RGB', ((image_width + separator_size) * cols - separator_size, image_height), "white")
        for idx, img in enumerate(row_images):
            stitched_row_image.paste(img, ((image_width + separator_size) * idx, 0))

        all_images.append(stitched_row_image)

    # Stitch together all rows into one final image
    final_image = Image.new('RGB', ((image_width + separator_size) * cols - separator_size, (image_height + separator_size) * rows - separator_size), "white")

    for idx, img in enumerate(all_images):
        final_image.paste(img, (0, (image_height + separator_size) * idx))

    # Save the final stitched image
    final_image.save("vis/final_image2.png")


# # ########################################################
# # ########                  METRIC                  ########
# # ########################################################
DO = False
if DO:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from math import sqrt
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import numpy as np

    def compute_metrics(model, X_test, y_test):
        # Make predictions
        y_pred = model.predict(X_test)

        # y_test = np.reshape(y_test, (y_test.shape[0]*y_test.shape[1]*y_test.shape[2], y_test.shape[3]))
        # y_pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1]*y_pred.shape[2], y_pred.shape[3]))
        from tensorflow.keras.losses import MeanAbsoluteError

        # assume that `predictions` and `ground_truth` are both of shape (N, 256, 256, 3)

        CLIP_MIN_S2, CLIP_MAX_S2 = 0, 3500
        y_pred = np.clip(y_pred, CLIP_MIN_S2, CLIP_MAX_S2)

        y_test = np.clip(y_test, CLIP_MIN_S2, CLIP_MAX_S2)

        y_pred = (y_pred - CLIP_MIN_S2) / (CLIP_MAX_S2 - CLIP_MIN_S2)
        y_test = (y_test - CLIP_MIN_S2) / (CLIP_MAX_S2 - CLIP_MIN_S2)

        mae_calculator = MeanAbsoluteError()
        mae = mae_calculator(y_test, y_pred).numpy()

        from tensorflow.keras.losses import MeanSquaredError

        # assume that `predictions` and `ground_truth` are both of shape (N, 256, 256, 3)
        mse_calculator = MeanSquaredError()
        rmse = sqrt(mse_calculator(y_test, y_pred).numpy())

        import tensorflow as tf

        def compute_psnr(img1, img2):
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return 100
            PIXEL_MAX = 1  # this is because your images are scaled to [0,1]
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

            # example usage:

        psnr_value = compute_psnr(y_pred, y_test)

        y_pred = y_pred.astype(np.float32)
        y_test = y_test.astype(np.float32)
        ssim_avg = tf.reduce_mean(tf.image.ssim(y_pred, y_test, max_val=1.0))

        return mae, rmse, ssim_avg, psnr_value

    n = 1
    model = load_model(models_bands[n][0])
    metric = compute_metrics(model, input_pred[:,:,:,:len(models_bands[n][1])+2], ground_truth)
    with open("metrics.txt", "a") as f:
        f.write(f"{models_bands[n][0]}\nMAE: {metric[0]}, RMSE: {metric[1]}, SSIM: {metric[2]}, PSNR: {metric[3]} \n")

# model = load_model(f"/Users/thomashebrard/thesis/code/project/pix2pix/models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=16,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=8,model_path=/model_epoch_320/model_epoch_320.h5")

# gen = model.predict(input_pred)
# predicted_array = ((gen + 1) / 2 * 400).astype(np.uint8)
# import numpy as np
# import matplotlib.pyplot as plt

# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming you have two arrays: 'predicted_array' and 'ground_truth_array',
# # both of shape (2, 256, 256, 4), representing the predicted and ground truth images respectively.

# # Iterate over the pictures
# import numpy as np
# import matplotlib.pyplot as plt

# # Assuming you have two arrays: 'predicted_array' and 'ground_truth_array',
# # both of shape (2, 256, 256, 4), representing the predicted and ground truth images respectively.


# for i in range(predicted_array.shape[0]):
#     # Iterate over the channels
#     for j in range(predicted_array.shape[3]):
#         # Extract the predicted and ground truth channels for the current picture
#         predicted_channel = ((predicted_array[i, :, :, j]+1)/2*250).astype(np.float32)
#         ground_truth_channel = ground_truth[i, :, :, j]
        
#         # Plot the predicted channel
#         plt.subplot(2, predicted_array.shape[3], i * predicted_array.shape[3] + j + 1)
#         plt.imshow(predicted_channel)  # Assuming grayscale channels, change cmap if needed
#         plt.axis('off')

#         # Plot the ground truth channel
#         plt.subplot(2, predicted_array.shape[3], (i+1) * predicted_array.shape[3] + j + 1)
#         plt.imshow(ground_truth_channel)  # Assuming grayscale channels, change cmap if needed
#         plt.axis('off')

# plt.show()

# B08_p = predicted_array[0,:,:,-1]
# B04_p = predicted_array[0, :,:,0]
# NDVI_p = (B08_p-B04_p)/(B08_p+B04_p)

# B08 = ground_truth[0,:,:,-1]
# B04 = ground_truth[0,:,:,0]
# NDVI = ((B08-B04)/(B08+B04))

# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# axes[0].imshow(NDVI, cmap='RdYlGn')
# axes[0].axis('off')
# axes[0].set_title('NDVI')

# # Plot the second image
# axes[1].imshow(NDVI_p, cmap='RdYlGn')
# axes[1].axis('off')
# axes[1].set_title('NDVI-p')
# plt.colorbar()

# # Adjust the spacing between subplots
# plt.tight_layout()

# # Display the plot
# plt.show()