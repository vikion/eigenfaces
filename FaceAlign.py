# code based on: https://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/

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
        face = self.face

        if self.left_eye is None or self.right_eye is None:
            self.M = None
            return

        left_eye_x, left_eye_y = self.left_eye
        right_eye_x, right_eye_y = self.right_eye

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
        face_rotated = [center[0] - face[2] // 2, center[1] - face[3] // 2, face[2], face[3]]

        x, y, w, h = face_rotated
        self.face = face_rotated
        self.img = rotated
        self.M = M

        # adjust features to the rotation

        if self.M is None:
            cv2.imwrite(fn, self.img)
            return

        face_rotated = self.face
        M = self.M
        scaling_factor = self.scaling_factor
        left_eye_center = self.left_eye
        right_eye_center = self.right_eye
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
        #cv2.circle(rotated, np.array([center_x, center_y]), 5, (50, 200, 50), 10)

class FaceAlign:

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def __init__(self, img_path, aspect_ratio=1.25, dpi=400):
        self.scaling_factor = 1.5
        self.ratio = aspect_ratio
    def process(self, path / Face):
        self. -> Face
        Face -> self.
        face = imread()
        self.crop_head(face)
        face.rotate()

        return cropped_face
    def crop_head(self, face):

        grayscale = cv2.cvtColor(face.img, cv2.COLOR_BGR2GRAY)
        boxes = FaceAlign.face_cascade.detectMultiScale(grayscale, 1.1, 4)
        box = max(boxes, key=lambda contour: contour[2] * contour[3])

        x, y, w, h = box
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
            face.left_eye = None
            face.right_eye = None
            return


        # select the two largest contours
        eyes = sorted(eyes, key=lambda contour: contour[2] * contour[3])[-2:]
        left_eye, right_eye = eyes if eyes[0][0] < eyes[1][0] else eyes[::-1]
        # get the centers of eyes
        left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
        right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)

        face.eyes['left'] = left_eye_center
        face.eyes['right'] = right_eye_center



    def crop(self):

        ratio = 1.125   # TODO should be object attribute

        # TODO maybe we'll want to scale-down width instead of scale-up height
        w = w
        h = int(ratio * w)

        centered_resized_face = (center_x - w // 2, center_y - h // 2, w, h)
        x, y, w, h = centered_resized_face

        #cv2.rectangle(rotated, (x, y), (x + w, y + h), (200, 15, 250), 3)
        plt.imshow(rotated)
        plt.show()

        w_px = 120
        h_px = int(w_px*ratio)

        cropped = rotated[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (w_px, h_px))

        return resized


if __name__ == "__main__":
    test_pictures = ["./kdmfi-kjp-ktvs-turany/photos/leginusova1-ktvs.jpg",
                     "./AIN extra/Frantisek Gyarfas.jpg"]
    for path in test_pictures:
        fn = path.split('/')[-1].strip(".jpg")
        print(fn)
        fa = FaceAlign(path)
        fa.crop_head()
        fa.detect_features()
        fa.rotate()
        fa.crop()

