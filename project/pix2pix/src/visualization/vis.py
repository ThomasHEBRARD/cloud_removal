import sys
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")

from src.training.models import Pix2Pix
from src.dataset.make_dataset import DataLoader

epoch = 10
model = load_model(f"models/model_epoch_{epoch}/model_epoch_{epoch}.h5")

dataset = DataLoader().load_batch(is_testing=True)[0]
input = np.array([dataset[1]])

s1_hv = dataset[1][:,:,0]
s1_vv = dataset[1][:,:,1]

s2_cloudy = dataset[1][:,:,2:]/2000
s2_cloud_free = dataset[0]/2000

gen_image = model.predict(input)

fig, axes = plt.subplots(3, 3, figsize=(16, 8))

axes[0, 0].imshow(s2_cloudy)
axes[0, 0].axis("off")
axes[0, 0].set_title("S2 RGB cloudy input")

axes[0, 1].imshow(s1_hv)
axes[0, 1].axis("off")
axes[0, 1].set_title("S1 VH input")

axes[0, 2].imshow(s1_vv)
axes[0, 2].axis("off")
axes[0, 2].set_title("S1 VV input")

# Plot output_image in the second block
axes[1, 1].imshow(gen_image[0])
axes[1, 1].axis("off")
axes[1, 1].set_title("Generated Output")

# Plot truth_image in the third block
axes[2, 1].imshow(s2_cloud_free)
axes[2, 1].axis("off")
axes[2, 1].set_title("Ground Truth")

# Remove unused subplots
axes[1, 0].remove()
axes[1, 2].remove()
axes[2, 0].remove()
axes[2, 2].remove()

plt.subplots_adjust(wspace=0.2, hspace=0.3)

fig.suptitle(f"Epoch : x/200, lr: 0.0002, gloss: x, dloss: ")
fig.savefig("test.png")
plt.close()
