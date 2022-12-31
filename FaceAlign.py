# code based on: https://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/

#  eyes 0.4 of face height
#     estimate based on: https://www.geeksforgeeks.org/ml-face-recognition-using-eigenfaces-pca-algorithm/ and physical ruler

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
        left_eye_x, left_eye_y = self.eyes['left']
        right_eye_x, right_eye_y = self.eyes['right']

        # calculate the rotation angle
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle = np.arctan(delta_y / delta_x)
        angle = (angle * 180) / np.pi

        im_h, im_w = self.img.shape[:2]
        center = (im_w // 2, im_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.img, M, (im_w, im_h))

        x, y, w, h = self.bbox_face
        dx, dy = (im_w - w) // 2, (im_h - h) // 2
        left_x, left_y = self.eyes['left']
        right_x, right_y = self.eyes['right']
        rotated_left = M @ (np.array([left_x, left_y, 1]) + np.array([dx, dy, 0]))
        rotated_left = rotated_left.astype(int)
        rotated_right = M @ (np.array([right_x, right_y, 1]) + np.array([dx, dy, 0]))
        rotated_right = rotated_right.astype(int)

        self.eyes['left'] = rotated_left
        self.eyes['right'] = rotated_right

        self.img = rotated
        self.M_rot = M


class FaceAlign:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

    def __init__(self, resize_w=1, resize_h=1.25):
        self.scaling_factor = 1.5
        self.resize_width = resize_w
        self.resize_height = resize_h

    def process(self, face):
        self.crop_head(face)
        if face.bbox_face is None:
            # cv2.imwrite("./zle/" + face.filename, face.img_original)

            # todo write warning
            return face.img
        # if not face.bboxface: return face.img; write to file

        self.detect_features(face)

        if face.eyes['left'] is not None and face.eyes['right'] is not None:
            face.rotate()
        else:
            # todo write warning
            pass

        self.center_crop(face)
        return face.img

    def crop_head(self, face):
        grayscale = cv2.cvtColor(face.img, cv2.COLOR_BGR2GRAY)
        boxes = FaceAlign.face_cascade.detectMultiScale(grayscale, 1.1, 4)

        # debug
        # for box in boxes:
        #    x, y, w, h = box
        #    cv2.rectangle(face.img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # plt.imshow(face.img)
        # plt.show()

        if not len(boxes):
            return

        box = max(boxes, key=lambda contour: contour[2] * contour[3])
        x, y, w, h = box

        # debug
        # cv2.rectangle(face.img, (x, y), (x + w, y + h), (0, 0, 200), 3)

        center_x, center_y = x + w // 2, y + h // 2
        W = int(w * self.scaling_factor)
        H = int(h * self.scaling_factor)
        X = center_x - W // 2
        Y = center_y - H // 2

        buffer_size = max(W, H)
        buffer_img = cv2.copyMakeBorder(face.img, buffer_size, buffer_size, buffer_size, buffer_size,
                                        cv2.BORDER_CONSTANT, value=0)

        X += buffer_size
        Y += buffer_size
        cropped_face = buffer_img[Y:Y + H, X:X + W]
        plt.imshow(cropped_face)
        plt.show()
        box[0] = (W - w) // 2
        box[1] = (H - h) // 2

        face.img = cropped_face
        face.bbox_face = box

    def detect_features(self, face):
        grayscale = cv2.cvtColor(face.img, cv2.COLOR_BGR2GRAY)
        x, y, w, h = face.bbox_face
        roi_gray = grayscale[y:(y + h), x:(x + w)]

        # TODO "optimize" the parameters of detectMultiScale
        # detect eyes
        eyes = FaceAlign.eye_cascade.detectMultiScale(roi_gray, 1.05, 4)

        # TODO alternative detection with "confidence"
        eyes3, rejectLevels, levelWeights = FaceAlign.eye_cascade.detectMultiScale3(roi_gray, scaleFactor=1.05, minNeighbors=4,
                                                                          outputRejectLevels=1)
        eyes = eyes3

        # select contours that are located in the upper half of the face
        eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if (h / 2) > (ey + eh/2) > (h / 5)]
        # TODO delete duplicates (eye in eye)

        # if len(eyes) < 2:
        #    eyes = FaceAlign.glasses_cascade.detectMultiScale(roi_gray, 1.1, 4)
        # select contours that are located in the upper half of the face
        #    eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey < (h // 2)]

        if len(eyes) < 2:
            return

        # debug
        roi_color = face.img[y:(y + h), x:(x + w)]
        for eye in eyes:
            xe, ye, we, he = eye
            cv2.rectangle(roi_color, (xe, ye), (xe + we, ye + he), (255, 0, 0), 3)
        #plt.imshow(face.img)
        #plt.show()

        # TODO selection by area
        # eyes = sorted(eyes, key=lambda contour: contour[2] * contour[3])[-2:]
        # left_eye, right_eye = eyes if eyes[0][0] < eyes[1][0] else eyes[::-1]

        # TODO selection by confidence
        eyes = [eye for _, eye in sorted(zip(levelWeights, eyes))][-2:]
        left_eye, right_eye = eyes if eyes[0][0] < eyes[1][0] else eyes[::-1]

        # TODO selection by estimate
        #im_w = face.img.shape[0]
        #f_w = w
        #shift = np.array([1, 1]) * (im_w - f_w) / 2
        #left_anticip = np.array([x + w / 4, y + h * 0.4]) - shift
        #right_anticip = np.array([x + 3 * w / 4, y + h * 0.4]) - shift

        #left_eye = min(eyes,
        #               key=lambda e: np.linalg.norm(left_anticip - np.array(e[0] + e[2] / 2, e[1] + e[3] / 2)))

        #right_eye = min(eyes,
        #                key=lambda e: np.linalg.norm(right_anticip - np.array(e[0] + e[2] / 2, e[1] + e[3] / 2)))


        # debug
        for eye in [left_eye, right_eye]:
            xe, ye, we, he = eye
            cv2.rectangle(roi_color, (xe, ye), (xe + we, ye + he), (5, 155, 180), 3)
        #plt.imshow(face.img)
        #plt.show()

        # get the centers of eyes
        left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
        right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)

        face.eyes['left'] = left_eye_center
        face.eyes['right'] = right_eye_center

    def center_crop(self, face):
        # try to detect face, if detected and eyes in detected: rewrite  -- (only if the img could be/was rotated)
        # else keep "old" face: centered in img center with old dims
        if face.eyes['left'] is not None and face.eyes['right'] is not None:
            grayscale = cv2.cvtColor(face.img, cv2.COLOR_BGR2GRAY)
            boxes = FaceAlign.face_cascade.detectMultiScale(grayscale, 1.1, 4)
            boxes_with_eyes = []
            left_eye, right_eye = face.eyes['left'], face.eyes['right']
            for box in boxes:
                left_in = (box[0] <= left_eye[0] <= box[0] + box[2] and
                           box[1] <= left_eye[1] <= box[1] + box[3])
                right_in = (box[0] <= right_eye[0] <= box[0] + box[2] and
                            box[1] <= right_eye[1] <= box[1] + box[3])
                if left_in and right_in:
                    boxes_with_eyes.append(box)
            if len(boxes):
                box = max(boxes, key=lambda contour: contour[2] * contour[3])
                face.bbox_face = box

        new_width = int(face.bbox_face[2] * self.resize_width)
        new_height = int(face.bbox_face[3] * self.resize_height)

        if face.eyes['left'] is not None and face.eyes['right'] is not None:
            # adjust features to the rotation
            left_x, right_x = face.eyes['left'][0], face.eyes['right'][0]
            center_x = (left_x + right_x) // 2

            # to keep width from face detection comment out next 3 lines
            e_prop_w = 2.125    # TODO should be object attribute
            new_width = int(abs(right_x - left_x) * e_prop_w)
            new_height = int(new_width * self.resize_height)

            eyes_y = face.eyes['left'][1]  # W.L.O.G left == right  (hopefully)
            e_prop_h = 0.4     # TODO should be object attribute
            center_y = eyes_y + new_height * (1 / 2 - e_prop_h)
        else:
            x, y, w, h = face.bbox_face
            center_x = x + w // 2
            center_y = y + h // 2

        new_x = int(center_x - new_width // 2)
        new_y = int(center_y - new_height // 2)
        centered_bbox = [new_x, new_y, new_width, new_height]
        face.bbox_face = centered_bbox

        # TODO should be object attribute
        w_px = 120
        h_px = int(w_px * self.resize_height)
        cropped = face.img[new_y:new_y + new_height, new_x:new_x + new_width]
        resized = cv2.resize(cropped, (w_px, h_px))
        face.img = resized


if __name__ == "__main__":
    test_pictures = ["./photos/KTVS_leginusova1-ktvs.jpg",
                     "./photos/AIN extra_Frantisek Gyarfas.jpg"]

    align = FaceAlign()
    DIR = "./photos/"
    test_pictures = os.listdir(DIR)
    # test_pictures = ['MAT_kollar53.jpg']
    for f in test_pictures:
        filename = f.strip('.jpg')
        print(filename, f)
        path = DIR + f
        face = Face(path)
        img = align.process(face)
        cv2.imwrite(f"./photos_aligned/run05all_loweyes/{filename}_align.jpg", img)
