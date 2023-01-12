# eyes alignment code based on: https://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/

import os
import cv2
import numpy as np

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

    def __init__(self, buffer=1.5, resize_w=1, resize_h=1.25, size_px=120, eyes2height=0.4, eyes2width=2.125):
        self.scaling_factor = buffer
        self.resize_width = resize_w
        self.resize_height = resize_h
        self.size_px = size_px
        self.eyes2height = eyes2height
        self.eyes2width = eyes2width

    def process(self, face):
        msg = True
        self.crop_head(face)
        if face.bbox_face is None:
            print(f"face not detected in {face.filename}")
            return face.img, False

        self.detect_features(face)

        if face.eyes['left'] is not None and face.eyes['right'] is not None:
            face.rotate()
        else:
            print(f"eyes not detected in {face.filename}")
            msg = False

        self.center_crop(face)
        return face.img, msg

    def crop_head(self, face):
        grayscale = cv2.cvtColor(face.img, cv2.COLOR_BGR2GRAY)
        boxes = FaceAlign.face_cascade.detectMultiScale(grayscale, 1.1, 4)

        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(face.img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if not len(boxes):
            return

        box = max(boxes, key=lambda contour: contour[2] * contour[3])
        x, y, w, h = box

        cv2.rectangle(face.img, (x, y), (x + w, y + h), (0, 0, 200), 3)

        center_x, center_y = x + w // 2, y + h // 2
        W = int(w * self.scaling_factor)
        H = int(h * self.scaling_factor)
        X = center_x - W // 2
        Y = center_y - H // 2

        buffer_size = max(W, H)
        buffer_img = cv2.copyMakeBorder(face.img, buffer_size, buffer_size, buffer_size, buffer_size,
                                        cv2.BORDER_CONSTANT, value=0)

        buffer_img_orig = cv2.copyMakeBorder(face.img_original, buffer_size, buffer_size, buffer_size, buffer_size,
                                        cv2.BORDER_CONSTANT, value=0)

        X += buffer_size
        Y += buffer_size
        cropped_face = buffer_img[Y:Y + H, X:X + W]
        cropped_face_orig = buffer_img_orig[Y:Y + H, X:X + W]

        box[0] = (W - w) // 2
        box[1] = (H - h) // 2

        face.img = cropped_face
        face.img_original = cropped_face_orig
        face.bbox_face = box

    def detect_features(self, face):
        def filter_duplicates(eyes):
            outer = []
            for eye1 in eyes:
                for eye2 in eyes:
                    if eye1 == eye2:
                        continue
                    if (eye1[0] <= eye2[0] and eye1[1] <= eye2[1] and
                            (eye1[0] + eye1[2]) >= (eye2[0] + eye2[2]) and (eye1[1] + eye1[3]) >= (eye2[1] + eye2[3])):
                        outer.append(eye1)
            for eye in outer:
                if eye in eyes:
                    eyes.remove(eye)
            return eyes

        image = face.img.copy()
        grayscale = cv2.cvtColor(face.img, cv2.COLOR_BGR2GRAY)

        x, y, w, h = face.bbox_face
        roi_gray = grayscale[y:(y + h), x:(x + w)]


        #  detection with "confidence"
        eyes3, rejectLevels, levelWeights = FaceAlign.eye_cascade.detectMultiScale3(roi_gray, scaleFactor=1.05, minNeighbors=5,
                                                                          outputRejectLevels=1)
        eyes = eyes3

        # select contours that are located in the upper half of the face
        eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if (h / 2) > (ey + eh/2) > (h / 5)]

        eyes = filter_duplicates(eyes)
        if len(eyes) < 2:
            eyes3, rejectLevels, levelWeights = FaceAlign.eye_cascade.detectMultiScale3(roi_gray, scaleFactor=1.015,
                                                                                        minNeighbors=3,
                                                                                        minSize=[5, 5],
                                                                                        outputRejectLevels=1)
            eyes = eyes3
            eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if (h / 2) > (ey + eh / 2) > (h / 5)]

        eyes = filter_duplicates(eyes)
        if len(eyes) < 2:
            grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
            roi_gray = grayscale[y:(y + h), x:(x + w)]
            eyes3, rejectLevels, levelWeights = FaceAlign.eye_cascade.detectMultiScale3(roi_gray, scaleFactor=1.004,
                                                                                        minNeighbors=2,
                                                                                        minSize=[1, 1],
                                                                                        outputRejectLevels=1)
            eyes = eyes3
            eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if (h / 2) > (ey + eh / 2) > (h / 5)]

        eyes = filter_duplicates(eyes)
        if len(eyes) < 2:
            face.img = face.img_original
            return

        #  selection with optimal d(eye1, eye2) approx. eq. face_size * r
        expected_eye_dist = w / 2.5
        pair = None
        error = 1e5
        for eye1 in eyes:
            for eye2 in eyes:
                if eye1 == eye2:
                    continue
                c1 = np.array([eye1[0] + eye1[2], eye1[1] + eye1[3]])
                c2 = np.array([eye2[0] + eye2[2], eye2[1] + eye2[3]])
                d = np.linalg.norm(c1 - c2)
                err = (expected_eye_dist - d)**2 + ((eye1[1] - eye2[1])/h)**2
                if err < error or pair is None:
                    pair = [eye1, eye2]
                    error = err

        left_eye, right_eye = pair if pair[0][0] < pair[1][0] else pair[::-1]


        # LESS SUCCESSFUL APPROACHES FOR EYE SELECTION
        #  selection by area
        #eyes = sorted(eyes, key=lambda contour: contour[2] * contour[3])[-2:]
        #left_eye, right_eye = eyes if eyes[0][0] < eyes[1][0] else eyes[::-1]

        #  selection by confidence
        #eyes = [eye for _, eye in sorted(zip(levelWeights, eyes))][-2:]
        # left_eye, right_eye = eyes if eyes[0][0] < eyes[1][0] else eyes[::-1]

        #  selection by estimate:  minimize the distance of anticipated eye location and detection for both left and right eye
        #im_w = face.img.shape[0]
        #f_w = w
        #shift = np.array([1, 1]) * (im_w - f_w) / 2
        #left_anticip = np.array([x + w / 4, y + h * 0.4]) - shift
        #right_anticip = np.array([x + 3 * w / 4, y + h * 0.4]) - shift
        #left_eye = min(eyes,
        #               key=lambda e: np.linalg.norm(left_anticip - np.array(e[0] + e[2] / 2, e[1] + e[3] / 2)))
        #right_eye = min(eyes,
        #                key=lambda e: np.linalg.norm(right_anticip - np.array(e[0] + e[2] / 2, e[1] + e[3] / 2)))


        # get the centers of eyes
        left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
        right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)

        face.eyes['left'] = left_eye_center
        face.eyes['right'] = right_eye_center
        face.img = face.img_original.copy()

    def center_crop(self, face):
        new_width = int(face.bbox_face[2] * self.resize_width)
        new_height = int(new_width * self.resize_height)

        if face.eyes['left'] is not None and face.eyes['right'] is not None:
            # adjust features to the rotation
            left_x, right_x = face.eyes['left'][0], face.eyes['right'][0]
            center_x = (left_x + right_x) // 2

            # to keep width from face detection comment out next 3 lines
            e_prop_w = self.eyes2width
            new_width = int(abs(right_x - left_x) * e_prop_w)

            new_height = int(new_width * self.resize_height)

            eyes_y = face.eyes['left'][1]  # W.L.O.G left == right  (hopefully)
            e_prop_h = self.eyes2height
            center_y = eyes_y + new_height * (1 / 2 - e_prop_h)
        else:
            x, y, w, h = face.bbox_face
            center_x = x + w // 2
            center_y = y + h // 2

        new_x = int(center_x - new_width // 2)
        new_y = int(center_y - new_height // 2)
        centered_bbox = [new_x, new_y, new_width, new_height]

        w_px = self.size_px
        h_px = int(w_px * self.resize_height)
        cropped = face.img[new_y:new_y + new_height, new_x:new_x + new_width]
        resized = cv2.resize(cropped, (w_px, h_px))
        face.img = resized


def eyes_from_canvas(event, x, y, flags, param):
    global face
    if event == cv2.EVENT_LBUTTONDOWN:
        w = face.img_original.shape[0]
        dw = int((w - face.bbox_face[2]) / 2)
        dw = np.array([dw, dw])
        refPt = np.array([x, y]) - dw

        if face.eyes['left'] is None:
            face.eyes['left'] = refPt
        elif face.eyes['right'] is None:
            face.eyes['right'] = refPt


if __name__ == "__main__":
    align = FaceAlign()
    DIR = "./photos/"
    test_pictures = os.listdir(DIR)
    detect_manually = []
    with open("manual_detection.txt") as f:
        for line in f:
            detect_manually.append(line.strip())

    manual_detection_stack = []
    for f in test_pictures:
        filename = f.strip('.jpg')
        #print(filename, f)
        path = DIR + f
        face = Face(path)
        img, msg = align.process(face)
        cv2.imwrite(f"./photos_aligned/run_3layers_lastblurred33/{filename}_align.jpg", img)

        if not msg or face.filename in detect_manually:
            face.eyes['left'] = face.eyes['right'] = None
            manual_detection_stack.append((path, face))

    for path, face in manual_detection_stack:
        img = face.img_original
        face.img = img.copy()
        cv2.namedWindow(path)
        cv2.setMouseCallback(path, eyes_from_canvas)
        while True:
            cv2.imshow(path, img)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                face.rotate()
                align.center_crop(face)
                cv2.imwrite(f"./photos_aligned/run_3layers_lastblurred33/{face.filename.strip('.jpg')}_align.jpg", face.img)
                break
        cv2.destroyAllWindows()
