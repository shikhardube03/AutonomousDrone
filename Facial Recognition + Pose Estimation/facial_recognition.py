from cvzone.FaceDetectionModule import FaceDetector
import cv2.cv2 as cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)
    height, width, channels = img.shape
    print(height, width, channels)

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        print(bboxs[0]["bbox"])
        # bbox - "y-min, x-min, height, width"
        bbox_list = bboxs[0]["bbox"]

        x_min = int((abs(bbox_list[1]) + bbox_list[1])/2)
        y_min = int((abs(bbox_list[0]) + bbox_list[0]) / 2)

        cropped_image = img[x_min:bbox_list[1] + bbox_list[3], y_min:bbox_list[0] + bbox_list[2]]
        cv2.imshow("cropped", cropped_image)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
