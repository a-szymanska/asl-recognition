from prepare import *
import cv2
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cap = cv2.VideoCapture(0)


def get_labels(img, no_pixels=64):
    kmeans = KMeans(n_clusters=2, n_init='auto')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    show_image(img, '?')
    img = cv2.resize(img, (no_pixels, no_pixels))
    X_flat = []
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            col = np.append(col, [i, j])
            X_flat.append(col)
    X_flat = np.array(X_flat)
    kmeans.fit(X_flat)
    labels = kmeans.predict(X_flat)
    if labels[0] == 1:
        labels = np.array(list(map(lambda c: 0 if c == 1 else 1, labels)))
    labels = [labels]
    return img, labels


if __name__ == '__main__':
    # Load model
    model = joblib.load('../trained_models/random_forest.pkl')

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            cv2.imshow('Camera', frame)
            key = cv2.waitKey(1)
            if key == ord('c'):
                break

        print('frame:', frame.shape)
        image, clustered_image = get_labels(frame)
        labels = model.predict(clustered_image)
        print(labels)
        show_image(image, chr(labels[0] + ord('A')))
