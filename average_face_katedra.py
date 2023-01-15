#https://leslietj.github.io/2020/06/28/How-to-Average-Images-Using-OpenCV/

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def compute_average():
    image_AIN = []
    image_hip = []
    image_INF = []
    image_KAFZM = []
    image_KDMFI = []
    image_KEF = []
    image_KJFB = []
    image_KJP = []
    image_KTF = []
    image_KTVS = []
    image_MAT = []
    folder_dir = "photos_aligned_cropped"
    for filename in os.listdir(folder_dir):
        # Read the input image
        img = cv2.imread("photos_aligned_cropped/" + filename, 1)
        if "KTVS" in filename:
            image_KTVS.append(img)
        elif "AIN" in filename:
            image_AIN.append(img)
        elif "hip" in filename:
            image_hip.append(img)
        elif "INF" in filename:
            image_INF.append(img)
        elif "KAFZM" in filename:
            image_KAFZM.append(img)
        elif "KDMFI" in filename:
            image_KDMFI.append(img)
        elif "KEF" in filename:
            image_KEF.append(img)
        elif "KJFB" in filename:
            image_KJFB.append(img)
        elif "KJP" in filename:
            image_KJP.append(img)
        elif "KTF" in filename:
            image_KTF.append(img)
        elif "MAT" in filename:
            image_MAT.append(img)

    ####KTVS
    avg_image_KTVS = image_KTVS[0]
    for i in range(len(image_KTVS)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_KTVS = cv2.addWeighted(image_KTVS[i], alpha, avg_image_KTVS, beta, 0.0)
    cv2.imwrite('average/avg_face_KTVS.png', avg_image_KTVS)

    ####AIN
    avg_image_AIN = image_AIN[0]
    for i in range(len(image_AIN)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_AIN = cv2.addWeighted(image_AIN[i], alpha, avg_image_AIN, beta, 0.0)
    cv2.imwrite('average/avg_face_AIN.png', avg_image_AIN)

    ####hip
    avg_image_hip = image_hip[0]
    for i in range(len(image_hip)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_hip = cv2.addWeighted(image_hip[i], alpha, avg_image_hip, beta, 0.0)
    cv2.imwrite('average/avg_face_hip.png', avg_image_hip)

    ####INF
    avg_image_INF = image_INF[0]
    for i in range(len(image_INF)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_INF = cv2.addWeighted(image_INF[i], alpha, avg_image_INF, beta, 0.0)
    cv2.imwrite('average/avg_face_INF.png', avg_image_INF)

    ####KAFZM
    avg_image_KAFZM = image_KAFZM[0]
    for i in range(len(image_KAFZM)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_KAFZM = cv2.addWeighted(image_KAFZM[i], alpha, avg_image_KAFZM, beta, 0.0)
    cv2.imwrite('average/avg_face_KAFZM.png', avg_image_KAFZM)

    ####KDMFI
    avg_image_KDMFI = image_KDMFI[0]
    for i in range(len(image_KDMFI)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_KDMFI = cv2.addWeighted(image_KDMFI[i], alpha, avg_image_KDMFI, beta, 0.0)
    cv2.imwrite('average/avg_face_KDMFI.png', avg_image_KDMFI)

    ####KEF
    avg_image_KEF = image_KEF[0]
    for i in range(len(image_KEF)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_KEF = cv2.addWeighted(image_KEF[i], alpha, avg_image_KEF, beta, 0.0)
    cv2.imwrite('average/avg_face_KEF.png', avg_image_KEF)

    ####KJFB
    avg_image_KJFB = image_KJFB[0]
    for i in range(len(image_KJFB)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_KJFB = cv2.addWeighted(image_KJFB[i], alpha, avg_image_KJFB, beta, 0.0)
    cv2.imwrite('average/avg_face_KJFB.png', avg_image_KJFB)

    ####KJP
    avg_image_KJP = image_KJP[0]
    for i in range(len(image_KJP)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_KJP = cv2.addWeighted(image_KJP[i], alpha, avg_image_KJP, beta, 0.0)
    cv2.imwrite('average/avg_face_KJP.png', avg_image_KJP)

    ####KTF
    avg_image_KTF = image_KTF[0]
    for i in range(len(image_KTF)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_KTF = cv2.addWeighted(image_KTF[i], alpha, avg_image_KTF, beta, 0.0)
    cv2.imwrite('average/avg_face_KTF.png', avg_image_KTF)

    ####MAT
    avg_image_MAT = image_MAT[0]
    for i in range(len(image_MAT)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image_MAT = cv2.addWeighted(image_MAT[i], alpha, avg_image_MAT, beta, 0.0)
    cv2.imwrite('average/avg_face_MAT.png', avg_image_MAT)


## https://www.tutorialspoint.com/how-to-compare-two-images-in-opencv-python
def compare(input):
    input = cv2.imread(input)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    h, w = input.shape

    ain = cv2.imread('average/avg_face_AIN.png')
    hip = cv2.imread('average/avg_face_hip.png')
    inf = cv2.imread('average/avg_face_INF.png')
    ktvs = cv2.imread('average/avg_face_KTVS.png')
    mat = cv2.imread('average/avg_face_MAT.png')
    kafzm = cv2.imread('average/avg_face_KAFZM.png')
    kdmfi = cv2.imread('average/avg_face_KDMFI.png')
    kef = cv2.imread('average/avg_face_KEF.png')
    kjfb = cv2.imread('average/avg_face_KJFB.png')
    kjp = cv2.imread('average/avg_face_KJP.png')
    ktf = cv2.imread('average/avg_face_KTF.png')

    ain = cv2.cvtColor(ain, cv2.COLOR_BGR2GRAY)
    hip = cv2.cvtColor(hip, cv2.COLOR_BGR2GRAY)
    inf = cv2.cvtColor(inf, cv2.COLOR_BGR2GRAY)
    ktvs = cv2.cvtColor(ktvs, cv2.COLOR_BGR2GRAY)
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    kafzm = cv2.cvtColor(kafzm, cv2.COLOR_BGR2GRAY)
    kdmfi = cv2.cvtColor(kdmfi, cv2.COLOR_BGR2GRAY)
    kef = cv2.cvtColor(kef, cv2.COLOR_BGR2GRAY)
    kjfb = cv2.cvtColor(kjfb, cv2.COLOR_BGR2GRAY)
    ktf = cv2.cvtColor(ktf, cv2.COLOR_BGR2GRAY)
    kjp = cv2.cvtColor(kjp, cv2.COLOR_BGR2GRAY)



    def error(imj, imk):
       diff = cv2.subtract(imj, imk)
       err = np.sum(diff**2)
       mse = err/(float(h*w))
       return mse

    score = {'AIN': error(input, ain),
             'hip': error(input, hip),
             'INF': error(input, inf),
             'KAFZM': error(input, kafzm),
             'KDMFI': error(input, kdmfi),
             'KEF': error(input, kef),
             'KJFB': error(input, kjfb),
             'KJP': error(input, kjp),
             'KTF': error(input, ktf),
             'KTVS': error(input, ktvs),
             'MAT': error(input, mat)
             }

    print(max(score, key= lambda x: score[x]))
#compute_average()
compare('photos_aligned_cropped/INF_Branislav Rovan_align.jpg')


