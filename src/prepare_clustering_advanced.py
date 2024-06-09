from prepare import *
from prepare_clustering import *
import numpy as np
import cv2
from sklearn.cluster import KMeans


img = cv2.imread("../data/asl_alphabet_test/A1.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
n = 64

hand_data = cv2.CascadeClassifier('../data/hand.xml')
hand_area = hand_data.detectMultiScale(img_rgb, minSize=(5, 5))

if len(hand_area) != 0:
    x, y, width, height = hand_area[0]
    margin = 10
    x_left, x_right = max(x - margin, 0), min(x + width + margin, img_rgb.shape[1])
    y_left, y_right = max(y - margin, 0), min(y + height + margin, img_rgb.shape[0])
    hand_img = img_rgb[y_left:y_right, x_left:x_right]
    hand_img = cv2.resize(hand_img, (n, n))

    # Prepare data
    X_flat = []
    for i, row in enumerate(hand_img):
        for j, x in enumerate(row):
            x = np.append(x, [i, j])
            X_flat.append(x)
    X_flat = np.array(X_flat)

    # Do clustering
    model = KMeans(n_clusters=2, n_init='auto', algorithm='lloyd')
    model.fit(X_flat)
    labels = model.predict(X_flat)
    if labels[0] == 1:
        labels = np.array(list(map(lambda c: 0 if c == 1 else 1, labels)))
    labels = labels.reshape((n, n))
    labels = cv2.resize(labels, (x_right-x_left, y_right-y_left), interpolation=cv2.INTER_NEAREST)

    # Map back
    labels_full = np.zeros(img.shape[0] * img.shape[1])
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            labels_full[(y_left + i) * img_rgb.shape[1] + (x_left + j)] = labels[i, j]

    # Just checking
    labels_img = [[255*l, 255*l, 255*l] for l in labels_full]
    img = np.reshape(labels_img, (200, 200, 3))
    show_image(img, "0")
else:
    print("No hands found")
