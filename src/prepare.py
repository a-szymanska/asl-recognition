import random
import numpy as np
import csv
from PIL import Image
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import os


def get_data(path, label):
    print('prepare', path)
    image = Image.open(path)
    image = image.convert("RGB")
    pixels = np.array(image.getdata())
    dataset = np.array([[pix[0], pix[1], pix[2], i % 200, i // 200] for i, pix in enumerate(pixels)])
    dataset_labeled = np.append(dataset, np.array([[label, label, label, label, label]]), axis=0)  # to be discussed
    return dataset_labeled


def read_data(n_files, path='../data/test.csv'):
    dataset = genfromtxt(path, delimiter=',')
    dataset = dataset.reshape((n_files, 40001, 5))
    X, y = dataset[:, :-1, :], dataset[:, -1, 0]
    return X, y


# train_size < 1 may be used for learning curve
def split_data(X, y, test_size=0.5, train_size=1):
    n_classes = int(np.max(y)) + 1
    classes = []
    for i in range(n_classes):
        mask = (y[:] == i)
        classes.append([X[mask], y[mask]])

    k = int(len(X) * test_size / n_classes)
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for i, cls in enumerate(classes):
        if k > len(cls):
            print(f'Too few observations from class {i}')
            return X_train, X_test, Y_train, Y_test

    for cls in classes:
        X_tr, X_tst, Y_tr, Y_tst = train_test_split(cls[0], cls[1], test_size=test_size)
        X_train.extend(X_tr)
        X_test.extend(X_tst)
        Y_train.extend(Y_tr)
        Y_test.extend(Y_tst)

    seed = random.randint(0, 100004)
    random.Random(seed).shuffle(X_train)
    random.Random(seed).shuffle(Y_train)
    seed = random.randint(0, 100004)
    random.Random(seed).shuffle(X_test)
    random.Random(seed).shuffle(Y_test)

    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':

    # Prepare and save data
    # directory = os.fsencode('../data/asl_alphabet_test')
    # with open('../data/test.csv', 'w', newline='\n') as file:
    #     writer = csv.writer(file)
    #     for file in os.listdir(directory):
    #         filename = os.fsdecode(file)
    #         if not filename.endswith(".jpg"):
    #             continue
    #         if filename[0].isupper():
    #             data = get_data(f'../data/asl_alphabet_test/{filename}', ord(filename[0]) - ord('A'))
    #         elif filename.startswith('space'):
    #             data = get_data(f'../data/asl_alphabet_test/{filename}', 26)
    #         data_reshaped = data.reshape(data.shape[0], -1)
    #         for row in data:
    #             writer.writerow(row)

    X, y = read_data(27)
    print(X.shape, y.shape)

    # X_train, X_test, Y_train, Y_test = split_data(X, y)
