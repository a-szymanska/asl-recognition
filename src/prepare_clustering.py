from prepare import *
import numpy as np
import os
from sklearn.cluster import KMeans


def get_clustered_data(no_files, filepath, X, Y, model):
    clustered_labels_list = []
    for idx, x in enumerate(X[:no_files]):
        print(f'{idx} out of {no_files}')
        X_flat = []
        for i, row in enumerate(x):
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
        filepath='data/train_clustered.npz',
        X=X_train, Y=Y_train,
        model=model
    )

    model = KMeans(n_clusters=2, n_init='auto')
    get_clustered_data(
        no_files=400,
        filepath='data/test_clustered.npz',
        X=X_test, Y=Y_test,
        model=model
    )
