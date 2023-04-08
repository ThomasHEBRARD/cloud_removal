import sys
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
sys.path.append(".")
sys.path.append("..")

from src.training.models import Pix2Pix

model = load_model('models/model_epoch_0.h5')
pix = Pix2Pix()
gen_image = model.predict(pix.dataset[1])

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

fig.suptitle(f"Epoch : x/200, lr: 0.0002, gloss: x, dloss: ")
fig.savefig("test.png")
plt.close()