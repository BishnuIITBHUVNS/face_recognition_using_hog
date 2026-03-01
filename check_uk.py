import os
import face_recognition
from PIL import Image
import numpy as np

KNOWN_FACES_DIR = "known_faces"

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    full_path = f"{KNOWN_FACES_DIR}/{name}"
    if os.path.isdir(full_path):
        for filename in os.listdir(full_path):
            img_path = f"{full_path}/{filename}"
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            # Find locations FIRST, then pass to face_encodings
            locations = face_recognition.face_locations(image, model="hog")
            print(f"{filename}: found {len(locations)} face(s)")
            encodings = face_recognition.face_encodings(image, locations)
            if len(encodings) > 0:
                known_faces.append(encodings[0])
                known_names.append(name)
                print(f"Successfully encoded {name}")