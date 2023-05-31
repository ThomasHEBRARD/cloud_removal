import sys
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from src.training.models import Pix2Pix
from src.dataset.make_dataset import DataLoader

epoch = 95
start = "run_2023-05-31T17:39:21"
model = load_model(f"models/{start}/model_epoch_{epoch}/model_epoch_{epoch}.h5")

BATCH_SIZE = 3

################################

savemode_data_loader = DataLoader(path="S1_32VNH_20200509_VV_402_538380_6274400_256")
bands = ["B04", "B03", "B02", "B08"]
ground_truth, input = zip(
    *savemode_data_loader.load_batch(
        bands=bands, batch_size=1, is_testing=True
    )
)

input = np.array(input)
input_pred = np.array([[d["data"] for d in inner_array] for inner_array in np.array(input)])
input_pred = np.transpose(input_pred, (0, 2, 3, 1))

# ground_truth = (((np.array(ground_truth) + 1) / 2) * 255).astype(np.uint8)
ground_truth = np.array([[d["data"] for d in inner_array] for inner_array in np.array(ground_truth)])
ground_truth = np.transpose(ground_truth, (0, 2, 3, 1))

output = model.predict(input_pred)
generated_image = ((output + 1) / 2 * 255).astype(np.uint8)

input_dict = {
    "s1_hv": {"title": "S1 HV", "desc": input[0][0]["im"].split("/")[-1], "image": input_pred[:, :, :, 0]},
    "s1_vv": {"title": "S1 VV", "desc": input[0][1]["im"].split("/")[-1], "image": input_pred[:, :, :, 1]},
}

for i in range(2, len(bands) + 2):
    input_dict[f"s2_{bands[i - 2]}"] = {
        "title": f"S2 {bands[i - 2]}",
        "desc": input[0][i]["im"].split("/")[-1],
        "image": (((input_pred[:, :, :, i] + 1) / 2) * 255).astype(np.uint8),
    }

#####################################
########     PLOT THE INPUT  ########
#####################################

for idx_img in range(1):
    num_images = len(bands) + 2
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
                    band = bands[idx - 2]
                    ax.imshow(input_dict[f"s2_{band}"]["image"][idx_img])
                    ax.set_title("_".join(input_dict[f"s2_{band}"]["desc"].split("_")[:5]))
                    ax.axis("off")
            else:
                (axes1[i, j] if num_rows > 1 else axes1[j]).axis("off")

    fig1.savefig(f"vis/input.png")

    with open(f"vis/input.txt", "a") as f:
        f.write(input_dict["s1_hv"]["desc"].split(".")[0] + "\n")
        f.write(input_dict["s1_vv"]["desc"].split(".")[0] + "\n")
        f.write(input_dict[f"s2_B02"]["desc"].split(".")[0] + "\n")

    #####################################
    ########     PLOT THE OUTPUT ########
    #####################################

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 8))

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

    fig2.savefig(f"vis/output.png")
    plt.close()
