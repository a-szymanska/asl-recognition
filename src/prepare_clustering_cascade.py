from prepare import *
import numpy as np
import cv2
from sklearn.cluster import KMeans
import os

hand_data = cv2.CascadeClassifier('data/hand.xml')


def get_clustered_data(no_files, filepath, X, Y, model):
    clustered_labels_list = []
    n = 64
    for idx, img in enumerate(X[:no_files]):
        print(f'{idx} out of {no_files}')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                for j, col in enumerate(row):
                    col = np.append(col, [i, j])
                    X_flat.append(col)
            X_flat = np.array(X_flat)

            # Do clustering
            model = KMeans(n_clusters=2, n_init='auto', algorithm='lloyd')
            model.fit(X_flat)
            labels = model.predict(X_flat)
            if labels[0] == 1:
                labels = np.array(list(map(lambda c: 0 if c == 1 else 1, labels)))
            labels = labels.reshape((n, n))
            labels = cv2.resize(labels, (x_right - x_left, y_right - y_left), interpolation=cv2.INTER_NEAREST)

            # Map back
            print(img.shape)
            all_labels = np.zeros(img.shape[0] * img.shape[1])
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    all_labels[(y_left + i) * img_rgb.shape[1] + (x_left + j)] = labels[i, j]
            labels = all_labels
        else:
            X_flat = []
            for i, row in enumerate(img):
                for j, col in enumerate(row):
                    col = np.append(col, [i, j])
                    X_flat.append(col)
            X_flat = np.array(X_flat)

            model.fit(X_flat)
            labels = model.predict(X_flat)
            if labels[0] == 1:
                labels = np.array(list(map(lambda c: 0 if c == 1 else 1, labels)))

        clustered_labels_list.append(labels)

    X_clustered = np.array(clustered_labels_list)
    file = os.path.join(project_dir, filepath)
    np.savez(file, X=X_clustered, Y=Y[:no_files])


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Cluster data and save labels
    train_data_file = os.path.join(project_dir, 'data/train3.npz')
    test_data_file = os.path.join(project_dir, 'data/test3.npz')

    X_train, Y_train = read_data(train_data_file)
    X_test, Y_test = read_data(test_data_file)

    model = KMeans(n_clusters=2, n_init='auto')
    get_clustered_data(
        no_files=50000,
        filepath='data/train_clustered_cascade.npz',
        X=X_train, Y=Y_train,
        model=model
    )

    model = KMeans(n_clusters=2, n_init='auto')
    get_clustered_data(
        no_files=400,
        filepath='data/test_clustered_cascade.npz',
        X=X_test, Y=Y_test,
        model=model
    )
