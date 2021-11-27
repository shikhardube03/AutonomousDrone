from cvzone.FaceDetectionModule import FaceDetector
import cv2.cv2 as cv2
import face_recognition
from PIL import Image

cap = cv2.VideoCapture(0)
detector = FaceDetector()

known_image = face_recognition.load_image_file("/Users/shikhar/Desktop/Group52-FA21/Facial Recognition + Pose Estimation/Faces/Shikhar2.jpg")
known_encoding  = face_recognition.face_encodings(known_image)[0]
#print(known_image)
#print(known_encoding)

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    height, width, channels = img.shape
    #print(height, width, channels)

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        # cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # print(bboxs[0]["bbox"])
        # bbox - "y-min, x-min, height, width"
        bbox_list = bboxs[0]["bbox"]

        x_min = int((abs(bbox_list[1]) + bbox_list[1])/2)
        y_min = int((abs(bbox_list[0]) + bbox_list[0]) / 2)

        cropped_image = img[x_min:bbox_list[1] + bbox_list[3], y_min:bbox_list[0] + bbox_list[2]]
        cropped_encoding = face_recognition.face_encodings(cropped_image)
        if(len(cropped_encoding) > 0):
            cropped_encoding = cropped_encoding[0]
            results = face_recognition.compare_faces([known_encoding], cropped_encoding)
            print(results)
        
        cv2.imshow("cropped", cropped_image)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
