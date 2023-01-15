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
U, S, VT = np.linalg.svd(A, full_matrices=0)
U.shape
face_diffs = dict()
for name, face in faces.items():
    face = cv2.blur(face,(5,5))
    face_vec = face.flatten()
    face_diffs[name] = face_vec - avg_face
face_vals = dict()
#modes = [5, 6, 7, 8]
modes = [6, 7]
for name, face in face_diffs.items():
    val = (np.ravel(U[:, modes].T @ face))
    face_vals[name] = val
pca = []
names_clustering = []
for name1, val1 in face_vals.items():
    best_dist = float("inf")
    pca.append(val1)
    names_clustering.append(name1)
    best_name = ""
    for name2, val2 in face_vals.items():
        if name1 != name2 and np.linalg.norm(val1 - val2) < best_dist:

            best_dist = np.linalg.norm(val1 - val2)
            best_name = name2
    #print("Najpodobnejsia tvar k", name1, "je", best_name, best_dist)
    print(f"{name1:42} {best_name:42} {best_dist:1.18}")

pca = np.array(pca)
print(pca.shape)
plt.scatter(x=pca[:, 0], y=pca[:, 1])
names = []
vals = []
for name, face in face_vals.items():
    names.append(name)
    vals.append(face)
iner = []
for i in range(1,30):
    kmeans = KMeans(n_clusters=i, random_state=3, algorithm="lloyd").fit(vals)
    iner.append(kmeans.inertia_)

plt.plot(iner)
for name1, val1 in face_vals.items():
    best_dist = float("inf")
    best_name = ""
    for name2, val2 in face_vals.items():
        if name1 != name2 and np.linalg.norm(val1 - val2) < best_dist:
            best_dist = np.linalg.norm(val1 - val2)
            best_name = name2
    # print("Najpodobnejsia tvar k", name1, "je", best_name, best_dist)
    print(f"{name1:42} {best_name:42} {best_dist:1.18}")

names = []
vals = []
for name, face in face_vals.items():
    names.append(name)
    vals.append(face)
iner = []
for i in range(1,30):
    kmeans = KMeans(n_clusters=i, random_state=3, algorithm="lloyd").fit(vals)
    iner.append(kmeans.inertia_)

plt.plot(iner)

for name1, val1 in face_vals.items():
    best_dist = float("inf")
    best_name = ""
    for name2, val2 in face_vals.items():
        if name1 != name2 and np.linalg.norm(val1 - val2) < best_dist:
            best_dist = np.linalg.norm(val1 - val2)
            best_name = name2
    # print("Najpodobnejsia tvar k", name1, "je", best_name, best_dist)
    print(f"{name1:42} {best_name:42} {best_dist:1.18}")
kmeans = KMeans(n_clusters=10, random_state=0, algorithm="lloyd").fit(pca)
plt.scatter(x=pca[:, 0], y=pca[:, 1], c=kmeans.labels_)


num_values = len(set(kmeans.labels_))

clustered = [[] for i in range(num_values)]

for cluster in range(len(kmeans.labels_)):

    clustered[kmeans.labels_[cluster]].append(names_clustering[cluster])




for cluster in clustered:

    fig = plt.figure(figsize=(10, 10))

    for j in range (len(cluster)):
        fig.add_subplot(int(len(cluster) / 6)+1, 6,j+1)
        plt.rcParams["figure.autolayout"] = True
        photo = cv2.imread("photos_aligned_cropped/"+cluster[j])

        lastname = (cluster[j].split("_")[1]).split()

        if len(lastname) == 1:
            lastname = lastname[0].lower()
        else:
            lastname = lastname[1].lower()
        lastname = ''.join([k for k in lastname if not k.isdigit()])
        plt.axis('off')
        plt.title(lastname)

        rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb, cmap = plt.cm.Spectral)
    plt.show()