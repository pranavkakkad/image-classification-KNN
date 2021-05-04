import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths

import argparse
import imutils
import os
import cv2


def image_to_feature_vector(image, size=(32,32)):
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image,bins=(8,8,8)):
    hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist=cv2.normalize(hist)

    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input data set")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help= "number of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="Number of jobs for knn distance")

args=vars(ap.parse_args())


print("Describing Images")
image_paths=list(paths.list_images(args["dataset"]))

rawImages = []
features = []
labels = []

for (i, image_path) in enumerate(image_paths):
    image = cv2.imread(image_path)
    label  = image_path.split(os.path.sep)[-1].split(".")[0]

    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

    # for i%1000 ==0:
    #     print("processed {}/{}".format(i,len(image_paths)))

rawImages=np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

print("pixel matrix: {:.2f}Mb".format(rawImages.nbytes/(1024*1000.0)))
print("feature matrix {:.2f}MB".format(features.nbytes/(1024*1000.0)))

(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels,
                                                      test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLables, testLAbels) = train_test_split(
    features, labels, test_size=0.25, random_state=42
)

print("Evaulating raw pixel accuracy")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainRI,trainRL)
acc = model.score(testRI,testRL)
print("raw pixel accruacy {:.2f}%".format(acc*100))


