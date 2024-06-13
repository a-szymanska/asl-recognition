from prepare_clustering import *
import joblib
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def get_labels(root_folder, no_pixels=64):
    kmeans = KMeans(n_clusters=2, n_init='auto')
    labels_list = []
    image_list = []
    for image_file in os.listdir(root_folder):
        image_path = os.path.join(root_folder, image_file)
        if os.path.isfile(image_path):
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_list.append(img_rgb)
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
            labels_list.append(labels)
    labels_list = np.array(StandardScaler().fit_transform(labels_list))
    return image_list, labels_list


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load('../trained_models/random_forest.pkl')
    print('Model loaded')

    # Check on testing set
    # test_labels = os.path.join(project_dir, 'data/test_clustered.npz')
    # X_test, Y_test = read_data(test_labels)
    # X_test_scaled = StandardScaler().fit_transform(X_test)
    # Y_pred = model.predict(X_test_scaled)
    # print(accuracy_score(Y_test, Y_pred))

    # Check for own images
    images_folder = os.path.join(project_dir, 'data/images/')
    images, clustered_images = get_labels(images_folder)
    labels = model.predict(clustered_images)
    print(labels)
    for img, lbl in zip(images, labels):
        show_image(img, chr(lbl + ord('A')))