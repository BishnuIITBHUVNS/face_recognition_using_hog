import os
import cv2
import face_recognition
from PIL import Image
import numpy as np

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
tolerance = 0.6
frame_thickness = 3
font_thickness = 2
Model = "hog"

print("loading known faces")
known_faces = []
known_names = []

print("processing known faces")
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = Image.open(f"{KNOWN_FACES_DIR}/{name}/{filename}").convert("RGB")
        image = np.array(image)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_faces.append(encodings[0])
            known_names.append(name)

print("processing unknown faces")
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = Image.open(f"{UNKNOWN_FACES_DIR}/{filename}").convert("RGB")
    image = np.array(image)
    locations = face_recognition.face_locations(image, model=Model)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance=tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"match found: {match}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), font_thickness)

    cv2.imshow(filename, image)
    cv2.waitKey(20000)