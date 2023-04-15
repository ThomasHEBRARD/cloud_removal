import json
import sys

sys.path.append(".")
sys.path.append("..")

from models import Pix2Pix

with open("config.json", "r") as file:
    config = json.load(file)

model = Pix2Pix(lr=config["lr"], gf=config["gf"], df=config["df"], train=True)

model.train(epochs=config["epochs"], batch_size=config["batch_size"])
