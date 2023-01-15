import cv2
import os

# get the path/directory
folder_dir = "AIN"
for images in os.listdir(folder_dir):
    # Read the input image
    img = cv2.imread("AIN/"+images)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        faces = img[y:y + h, x:x + w]
        cv2.imwrite('cropped_hip/' + images, faces)
