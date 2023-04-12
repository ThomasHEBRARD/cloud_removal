import sys
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from src.training.models import Pix2Pix
from src.dataset.make_dataset import DataLoader

model = load_model("models/model_epoch_0/model_epoch_0.h5")
pix = Pix2Pix()
data = DataLoader().generator_data()
gen_image = model.predict(data[1])

fig, axes = plt.subplots(3, 3, figsize=(16, 8))

axes[0, 0].imshow(data[1][0][:, :, 2:])
axes[0, 0].axis("off")
axes[0, 0].set_title("S2 RGB cloudy input")

axes[0, 1].imshow(data[1][0][:, :, 0])
axes[0, 1].axis("off")
axes[0, 1].set_title("S1 VH input")

axes[0, 2].imshow(data[1][0][:, :, 1])
axes[0, 2].axis("off")
axes[0, 2].set_title("S1 VV input")

# Plot output_image in the second block
axes[1, 1].imshow(gen_image[0])
axes[1, 1].axis("off")
axes[1, 1].set_title("Generated Output")

# Plot truth_image in the third block
axes[2, 1].imshow(data[0][0])
axes[2, 1].axis("off")
axes[2, 1].set_title("Ground Truth")

# Remove unused subplots
axes[1, 0].remove()
axes[1, 2].remove()
axes[2, 0].remove()
axes[2, 2].remove()

plt.subplots_adjust(wspace=0.2, hspace=0.3)

fig.suptitle(f"Epoch : x/200, lr: 0.0002, gloss: x, dloss: ")
fig.savefig("test2.png")
plt.close()
