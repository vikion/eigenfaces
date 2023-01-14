import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.datasets
from sklearn.decomposition import PCA, TruncatedSVD
import seaborn as sns


_, _, files = list(os.walk("photos64"))[0]

faces = dict()

for f in files:
    f.encode('unicode_escape')
    img = cv2.imread("photos64/"+f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces[f] = gray

images = sklearn.datasets.fetch_olivetti_faces()["images"]
avg_face = np.zeros(64*64)
for face in images:
    face_vec = face.flatten()
    #face_vec = face_vec.reshape(face_vec.shape[0], 1)
    avg_face += face_vec

avg_face = avg_face/len(images)
avg_face_img = avg_face.reshape(64, 64)
face_train_diffs = list()
for face in images:
    face_vec = face.flatten()
    face_train_diffs.append(face_vec - avg_face)

A = np.matrix([i for i in face_train_diffs]).T
AtA = np.matmul(A.T, A)
_, vec = np.linalg.eigh(AtA)
eigen = np.array([np.array(np.matmul(A, v.T).T)[0] for v in vec]).T

face_diffs = dict()
for name, face in faces.items():
    face_vec = face.flatten()
    face_diffs[name] = face_vec - avg_face

face_vals = dict()
for name, face in face_diffs.items():
    face_vals[name] = np.linalg.lstsq(eigen, face, rcond=None)[0]

names = []
vals = []
for name, face in face_vals.items():
    names.append(name)
    vals.append(face)

iner = []
for i in range(1,30):
    kmeans = KMeans(n_clusters=i, random_state=3, algorithm="lloyd").fit(vals)
    iner.append(kmeans.inertia_)


last_names = []
couples= []
for i in names:

    lastname =i.split("_")[1]
    lastname  = lastname .split()
    if len(lastname ) ==1 :
            lastname  = lastname [0].lower()
    else:
            lastname  = lastname [1].lower()

    lastname  = ''.join([k for k in lastname  if not k.isdigit()])
    last_names.append(lastname )
    if(lastname +"ova" in last_names ):
        couples.append((lastname , lastname +"ova"))
    if lastname [:-3]  in last_names and lastname [-3::]== "ova":
        couples.append((lastname , lastname [:-3]))

print(couples)

import statistics

couples = [("KAFZM_Ladislav Meri_align.jpg", "AIN_Maria Meriova_align.jpg"),
           ("MAT_jajcay6_align.jpg","AIN_Tatiana Jajcayova_align.jpg"),
           ("AIN_Tomas Vinar_align.jpg", "INF_Bronislava Brejova_align.jpg"),
           ("KAFZM_Jozef Kristek DrSc_align.jpg","KAFZM_Miriam Kristekova PhD_align.jpg"),
           ("KEF_Miroslav Zahoran CSc_align.jpg" ,"KEF_Anna Zahoranova PhD_align.jpg"),
           ("KEF_Peter Markos DrSc_align.jpg"), ("AIN_Maria Markosova_align.jpg"),
           ("KJFB_Pavol Bartos PhD_align.jpg","hip_bartosova2_align.jpg")]


vals = []
vals_couple= []
for name1, val1 in face_vals.items():


    best_dist = float("inf")
    best_name = ""
    for name2, val2 in face_vals.items():

            vals.append(np.linalg.norm(val1 - val2))
            if((name1,name2) in couples):
                vals_couple.append(np.linalg.norm(val1 - val2))
                print(np.linalg.norm(val1 - val2), (name1,name2))


med = statistics.mode(vals)
med_couples =  statistics.mode(vals_couple[:-1])


import scipy
iqr = scipy.stats.iqr(vals)
q3, q1 = np.percentile(vals, [75, 25])
iqr = q3 - q1



plt.scatter(vals_couple,vals_couple,c="red")
y = [i for i in range(1,800)]
plt.plot([q1 - (1.5 * iqr)]*len(y), y)
y = [i for i in range(1,800)]
plt.plot([q3   + (1.5 * iqr)]*len(y), y)
plt.show()


