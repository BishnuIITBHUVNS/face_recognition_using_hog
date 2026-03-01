import os
from PIL import Image
import numpy as np

KNOWN_FACES_DIR = "known_faces"

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        path = f"{KNOWN_FACES_DIR}/{name}/{filename}"
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        print(f"{filename}: shape={image.shape}, dtype={image.dtype}")