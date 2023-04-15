import sys
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from src.training.models import Pix2Pix
from src.dataset.make_dataset import DataLoader

epoch = 100
model = load_model(f"models/model_epoch_{epoch}/model_epoch_{epoch}.h5")

BATCH_SIZE = 3

fig, axes = plt.subplots(BATCH_SIZE, 5, figsize=(20, 12))

dA, dB = zip(*DataLoader().load_batch(batch_size=BATCH_SIZE, is_testing=True))
dB = np.array(dB)
dA = np.array(dA)

input = dB

s1_hv = dB[:, :, :, 0]
s1_vv = dB[:, :, :, 1]

gen_image = model.predict(input)

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


fig.suptitle(f"Epoch : {epoch}/200, lr: 0.0002, gloss: x, dloss: ")
fig.savefig("vis/adhoc_vis.png")
plt.close()
