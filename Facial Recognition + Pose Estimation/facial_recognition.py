from cv2 import VideoCapture
from cvzone.FaceDetectionModule import FaceDetector
import cv2.cv2 as cv2
import numpy as np
import face_recognition
from PIL import Image


cap = cv2.VideoCapture(0)
detector = FaceDetector()

# loading images into workspace 
known_image = face_recognition.load_image_file("/Users/shikhar/Desktop/Group52-FA21/Facial Recognition + Pose Estimation/Faces/Shikhar2.jpg")
Shikhar_encoding  = face_recognition.face_encodings(known_image)[0]

known_image2 = face_recognition.load_image_file("/Users/shikhar/Desktop/Group52-FA21/Facial Recognition + Pose Estimation/Faces/Robert.jpg")
Robert_encoding = face_recognition.face_encodings(known_image2)[0]

# making array for known encodings and their names 
known_encodings = [Shikhar_encoding, Robert_encoding]
names = ["Shikhar", "Robert"]

# # Making local Variables to keep track of necessary info
# faceLocations = []
# faceEncodings = []
# faceName = []
# FrameActive = True

# while True:
#     ret, frame = cap.read()
#     smallFrame = cv2.resize(frame, (0,0), fx = 0.25, fy = 0.25)
#     rgbSmallFrame = smallFrame[:, :, ::-1]

#     if FrameActive:
#         faceLocations = face_recognition.face_locations(rgbSmallFrame)
#         faceEncodings = face_recognition.face_encodings(rgbSmallFrame, faceLocations)

#         faceNames = []
#         for singleFaceEncoding in faceEncodings:
#             matches = face_recognition.compare_faces(known_encodings, singleFaceEncoding)
#             name = "unknown"

#             # looking for first match and indentifying it if avaliable 
#             """ if True in matches:
#                 first_match_index = matches.index(True)
#                 name = faceNames[first_match_index] """
#             # finding closest face    
#             faceDistances = face_recognition.face_distance(known_encodings, singleFaceEncoding)
#             bestMatchindex = np.argmin(faceDistances)
#             if matches[bestMatchindex]:
#                 name = faceNames[bestMatchindex]
        
#             faceNames.append(name)
#     FrameActive = not FrameActive

#     for (top, right, bottom, left), name in zip(faceLocations, faceNames):
#         #resacling image 
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         # drawing boxes around name 
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # labeling name 
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#     cv2.imshow("Image", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    height, width, channels = img.shape
    #print(height, width, channels)

    if bboxs:
        for face in bboxs:
            bbox = face["bbox"]
            center = face["center"]

            x_min = int((abs(bbox[1]) + bbox[1])/2)
            y_min = int((abs(bbox[0]) + bbox[0]) / 2)
            cropped_image = img[x_min:bbox[1] + bbox[3], y_min:bbox[0] + bbox[2]]
            cropped_encoding = face_recognition.face_encodings(cropped_image)
            if(len(cropped_encoding) > 0):
                cropped_encoding = cropped_encoding[0]
                for x in range(0, len(known_encodings)):
                    results = face_recognition.compare_faces([known_encodings[x]], cropped_encoding)
                    print(results)
                    print(x)
                    if (not results[0]):
                        print("test")
                        continue;
                    else:
                        print(names[x])
                        cv2.putText(img, names[x],
                                (bbox[0], bbox[1] + 40), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)

        """ # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        #cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
        

        # bbox - "y-min, x-min, height, width"
        bbox = bboxs[0]["bbox"]
        print(bboxs[1])

        x_min = int((abs(bbox[1]) + bbox[1])/2)
        y_min = int((abs(bbox[0]) + bbox[0]) / 2)

        cropped_image = img[x_min:bbox[1] + bbox[3], y_min:bbox[0] + bbox[2]]
        cropped_encoding = face_recognition.face_encodings(cropped_image)
        if(len(cropped_encoding) > 0):
            cropped_encoding = cropped_encoding[0]
            results = face_recognition.compare_faces([Shikhar_encoding], cropped_encoding)
            cv2.putText(img, 'Shikhar',
                                (bbox[0], bbox[1] + 40), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 255, 0), 2)
            print(results)
        
        cv2.imshow("cropped", cropped_image) """

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
