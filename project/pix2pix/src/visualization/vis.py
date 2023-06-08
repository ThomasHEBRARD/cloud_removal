import sys
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from src.training.models import Pix2Pix
from src.dataset.make_dataset import DataLoader

epoch = 190
start = "epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path="
model = load_model(f"/Users/thomashebrard/thesis/code/project/cnn/m/model-100.h5")

models_bands = [
    (f"/Users/thomashebrard/thesis/code/project/cnn/m/model-100.h5", ["B04", "B03", "B02", "B08"], "unet"),
    (f"models/epochs=500,lr=0.0001,gf=64,df=64,batch_size=20,bands=['B04', 'B03', 'B02', 'B08'],nb_batches_per_epoch=10,model_path=/model_epoch_{475}/model_epoch_{475}.h5", ["B04", "B03", "B02", "B08"], "gan"),
    (f"models/DO_NOT_DELETE/model_epoch_{190}/model_epoch_{190}.h5", ["B04", "B03", "B02"], "first_gan")
]

################################
BATCH_SIZE = 10
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

########################################################
########                  PLOT                  ########
########################################################
rows = BATCH_SIZE
cols = 3 + len(models_bands)  # For cloudy_input, ground_truth and each model output
# fig, axes = plt.subplots(rows, cols, figsize=(16 * cols, 8 * rows))  # Adjust the figure size accordingly

# for image_idx in range(BATCH_SIZE):

#     # Create the figure and axes array

#     cloudy_input = np.stack(
#         (
#             input_dict[f"s2_B04"]["image"][image_idx],
#             input_dict[f"s2_B03"]["image"][image_idx],
#             input_dict[f"s2_B02"]["image"][image_idx],
#         ),
#         axis=-1,
#     )
#     # Display cloudy_input
#     axes[image_idx, 0].imshow(cloudy_input)
#     axes[image_idx, 0].set_title(f"Sentinel-2 Cloudy Input")
#     axes[image_idx, 0].axis("off")

#     # Display ground_truth
#     axes[image_idx, 1].imshow(ground_truth[image_idx])
#     axes[image_idx, 1].set_title(f"Ground Truth")
#     axes[image_idx, 1].axis("off")

#     # Display model outputs

#     for j, (model, bands, filename) in enumerate(models_bands):
#         print(f"MODEL: {model}", len(bands)+2, input_pred.shape)
#         model = load_model(model)
#         input_to_show = input_pred[:,:,:,:len(bands)+2]

#         output = model.predict(input_to_show)
#         generated_image = ((output + 1) / 2 * 255).astype(np.uint8)

#         axes[image_idx, j + 2].imshow(generated_image[image_idx])
#         axes[image_idx, j + 2].set_title(f"Model {j+1} Output")
#         axes[image_idx, j + 2].axis("off")

# plt.tight_layout()
# fig.savefig(f"vis/visualisation.png")  # Save the composite figure
# plt.close()

from PIL import Image

# Define the image dimensions and separator size
image_width, image_height = 256, 256  # Replace with your actual image size
separator_size = 10  # Size of white separator between images in pixels

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
        generated_image = ((output + 1) / 2 * 255).astype(np.uint8)

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
