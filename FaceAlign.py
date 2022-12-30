# code based on: https://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Face:

    def __init__(self, img_path):
        self.filename = img_path.split('/')[-1]
        self.img_original = cv2.imread(img_path)
        self.img = self.img_original.copy()
        self.eyes = {"left": None, "right": None}
        self.bbox_face = None
        self.rot_img = None
        self.M_rot = None

    def rotate(self):
        cropped_face = self.img
        face = self.bbox_face


        # todo
        if self.eyes['left'] is None or self.eyes['right'] is None:
            self.M = None
            return

        left_eye_x, left_eye_y = self.eyes['left']
        right_eye_x, right_eye_y = self.eyes['right']

        if left_eye_y > right_eye_y:
            A = right_eye_x, left_eye_y
            direction = -1
        else:
            A = left_eye_x, right_eye_y
            direction = 1

        # calculate the rotation angle
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle = np.arctan(delta_y / delta_x)
        angle = (angle * 180) / np.pi

        h, w = cropped_face.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(cropped_face, M, (w, h))
        #face_rotated = [center[0] - face[2] // 2, center[1] - face[3] // 2, face[2], face[3]]

        # detect face in rotated picture
        grayscale = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        plt.imshow(grayscale)
        plt.show()
        faces = FaceAlign.face_cascade.detectMultiScale(grayscale, 1.1, 4)
        face_rotated = max(faces, key=lambda contour: contour[2] * contour[3])

        x, y, w, h = face_rotated
        self.face = face_rotated
        self.img = rotated
        self.M = M

        # adjust features to the rotation

        if self.M is None:
            return

        face_rotated = self.face
        M = self.M
        scaling_factor = 1.5 # self.scaling_factor
        left_eye_center = self.eyes['left']
        right_eye_center = self.eyes['right']
        rotated = self.img

        x, y, w, h = face_rotated
        dx, dy = int(w * (scaling_factor - 1) // 2), int(h * (scaling_factor - 1) // 2)

        rotated_left = M @ (np.array([left_eye_center[0], left_eye_center[1], 1]) + np.array([dx, dy, 0]))
        rotated_left = rotated_left.astype(int)
        rotated_right = M @ (np.array([right_eye_center[0], right_eye_center[1], 1]) + np.array([dx, dy, 0]))
        rotated_right = rotated_right.astype(int)

        #cv2.circle(rotated, rotated_left, 5, (200, 200, 200), 10)
        #cv2.circle(rotated, rotated_right, 5, (200, 199, 199), 10)


        center_x = (rotated_left[0] + rotated_right[0]) // 2
        center_y = y + h // 2

        centered_face = [center_x - w // 2, center_y - h // 2, w, h ]
        self.bbox_face = centered_face
        #cv2.circle(rotated, np.array([center_x, center_y]), 5, (50, 200, 50), 10)

class FaceAlign:

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    def __init__(self, img_path, aspect_ratio=1.25, dpi=400):
        self.scaling_factor = 1.5
        self.ratio = aspect_ratio
    def process(self, face):
        if not self.crop_head(face):
            cv2.imwrite("./zle/" + face.filename, face.img_original)
            return face.img
        if not self.detect_features(face):
            cv2.imwrite("./zle/" + face.filename, face.img_original)
        face.rotate()
        self.crop(face)
        return face.img

    def crop_head(self, face):
        grayscale = cv2.cvtColor(face.img, cv2.COLOR_BGR2GRAY)
        boxes = FaceAlign.face_cascade.detectMultiScale(grayscale, 1.1, 4)
        for box in boxes:
            x, y, w, h = box
            #cv2.rectangle(face.img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print(boxes)
        if not len(boxes):

            return False
        box = max(boxes, key=lambda contour: contour[2] * contour[3])
        x, y, w, h = box

        #cv2.rectangle(face.img, (x, y), (x + w, y + h), (0, 0, 200), 3)

        center_x, center_y = x + w // 2, y + h // 2

        W = int(w * self.scaling_factor)
        H = int(h * self.scaling_factor)
        X = center_x - W // 2
        Y = center_y - H // 2

        pic_height, pic_width = face.img.shape[:2]
        if X > 0 and Y > 0 and X+W < pic_width and Y+H < pic_height:
            cropped_face = face.img[Y:Y + H, X:X + W]
            box[0] = (W - w) // 2
            box[1] = (H - h) // 2
        else:
            cropped_face = face.img

        face.img = cropped_face
        face.bbox_face = box

        return True


    def detect_features(self, face):
        x, y, w, h = face.bbox_face
        cropped_face = face.img
        grayscale = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        roi_gray = grayscale[y:(y + h), x:(x + w)]
        roi_color = cropped_face[y:(y + h), x:(x + w)]

        # detect eyes
        eyes = FaceAlign.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        # select contours that are located in the upper half of the face
        eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey < (h // 2)]

        if len(eyes) < 2:
            eyes = FaceAlign.glasses_cascade.detectMultiScale(roi_gray, 1.1, 4)
            # select contours that are located in the upper half of the face
            eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey < (h // 2)]

        # TODO
        if len(eyes) < 2:
            face.left_eye = None
            face.right_eye = None
            return False

        for eye in eyes:
            x, y, w, h = eye
            #cv2.rectangle(roi_color, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # select the two largest contours
        eyes = sorted(eyes, key=lambda contour: contour[2] * contour[3])[-2:]
        left_eye, right_eye = eyes if eyes[0][0] < eyes[1][0] else eyes[::-1]
        # get the centers of eyes
        left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
        right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)

        face.eyes['left'] = left_eye_center
        face.eyes['right'] = right_eye_center

        return True


    def crop(self, face):

        ratio = 1.125   # TODO should be object attribute

        x, y, w, h = face.bbox_face
        # TODO maybe we'll want to scale-down width instead of scale-up height
        new_w = w
        new_h = int(ratio * w)

        center_x, center_y = x + w//2, y + h//2
        resized_face = [center_x - new_w //2 , center_y - new_h //2, new_w, new_h]
        x, y, w, h = resized_face


        # TODO should be object attribute
        w_px = 120
        h_px = int(w_px*ratio)


        cropped = face.img[y:y+h, x:x+w]
        print(resized_face)
        #plt.imshow(face.img)
        #plt.imshow(cropped)
        #plt.show()
        #plt.imshow(cropped)
        resized = cv2.resize(cropped, (w_px, h_px))

        face.bbox_face = resized_face
        face.img = resized


if __name__ == "__main__":
    test_pictures = ["./photos/KTVS_leginusova1-ktvs.jpg",
                     "./photos/AIN extra_Frantisek Gyarfas.jpg"]
    test_pictures = glob.glob(f"./photos/*")
    align = FaceAlign(...)
    DIR = "./photos/"
    test_pictures = os.listdir(DIR)
    for f in test_pictures:

        filename = f.strip('.jpg')
        print(filename, f)
        path = DIR + f
        face = Face(path)
        img = align.process(face)
        cv2.imwrite(f"./photos_aligned/{filename}_align.jpg", img)

