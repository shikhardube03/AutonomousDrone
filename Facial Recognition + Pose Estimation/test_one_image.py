import face_recognition
from PIL import Image
import os
import sys


known_image = face_recognition.load_image_file("/Facial Recognition + Pose Estimation/Faces/Biden.jpeg")
unknown_image = face_recognition.load_image_file("/Facial Recognition + Pose Estimation/Faces/BidenTest.jpg")
print(known_image)

biden_encoding = face_recognition.face_encodings(known_image)[0]
print(biden_encoding)
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

print(results)