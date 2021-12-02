# install opencv "pip install opencv-python"
import cv2

# focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    # finding the focal length
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


# distance estimation function
def Distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame

    # return the distance
    return distance


def face_data(image):
    # face detector object
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    face_width = 0  # making face width to zero

    # converting color image ot gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)

    # looping through the faces detect in the image
    # getting coordinates x, y , width and height
    for (x, y, h, w) in faces:
        # getting face width in the pixels
        face_width = w
    # return the face width in pixel
    return face_width

def distance_calc(image):
    img = cv2.imread(image)

    # distance from camera to object(face) measured
    # centimeter
    known_distance = 76.2
    # width of face in the real world or Object Plane
    # centimeter
    known_width = 14.3

    # reading reference_image from directory
    ref_image = cv2.imread("Ref_image.png")

    # find the face width(pixels) in the reference_image
    ref_image_face_width = face_data(ref_image)

    # get the focal by calling "Focal_Length_Finder"
    # face width in reference(pixels),
    # Known_distance(centimeters),
    # known_width(centimeters)
    focal_length_found = Focal_Length_Finder(
        known_distance, known_width, ref_image_face_width)

    h, w, c = img.shape

    distance = Distance_finder(
        focal_length_found, known_width, w)
    return distance
