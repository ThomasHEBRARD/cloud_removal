import json
import sys

sys.path.append(".")
sys.path.append("..")

from models import Pix2Pix

with open("config.json", "r") as file:
    config = json.load(file)

model = Pix2Pix(
    bands=config["bands"], lr=config["lr"], gf=config["gf"], df=config["df"], train=True
)

model.train(
    epochs=config["epochs"],
    nb_batches_per_epoch=config["nb_batches_per_epoch"],
    batch_size=config["batch_size"],
    model_path=config["model_path"]
)
