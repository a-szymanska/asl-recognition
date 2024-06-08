import random
import numpy as np
import csv
import cv2
from PIL import Image
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_data(root_folder,file_path,no_pixels=32,no_items=3000):
    images = []
    labels = []
    for letter_folder in os.listdir(root_folder):
        letter_folder_path = os.path.join(root_folder, letter_folder)
        if os.path.isdir(letter_folder_path):
            if letter_folder.isupper():
                label = ord(letter_folder) - ord('A')
                for image_file in os.listdir(letter_folder_path)[:no_items]:
                    image_path = os.path.join(letter_folder_path, image_file)
                    if os.path.isfile(image_path):
                        image = cv2.imread(image_path)
                        image = cv2.resize(image, (no_pixels, no_pixels))
                        images.append(image)
                        labels.append(label)
        print(f'saved {letter_folder}')
    
    images = np.array(images)
    labels = np.array(labels)
    np.savez(file_path, images=images, labels=labels)
    print("saved!")



def read_data(path):
    data = np.load(path)
    images = data['images']
    labels = data['labels']
    data.close()
    return images,labels

def show_image(image,label):
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.axis('off')  # Hide axes
    plt.show()


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
    # alph = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['space']
    # for c in alph:
    #     directory = os.fsencode('../data/asl_alphabet_test/{c}')
    #     with open('../data/train.csv', 'w', newline='\n') as file:
    #         writer = csv.writer(file)
    #         for file in os.listdir(directory):
    #             filename = os.fsdecode(file)
    #             if not filename.endswith(".jpg"):
    #                 continue
    #             if filename[0].isupper():
    #                 data = get_data(f'../data/asl_alphabet_test/{filename}', ord(filename[0]) - ord('A'))
    #             elif filename.startswith('space'):
    #                 data = get_data(f'../data/asl_alphabet_test/{filename}', 26)
    #             data_reshaped = data.reshape(data.shape[0], -1)
    #             for row in data:
    #                 writer.writerow(row)
    data_file=os.path.join(project_dir,'data/train3.npz')
    images_folder=os.path.join(project_dir, 'data/archive/asl_alphabet_train/asl_alphabet_train/')
    #save_data(images_folder,data_file,no_pixels=200,no_items=200)
    X,Y=read_data(data_file)
    show_image(X[213],Y[213])
    # X_train, X_test, Y_train, Y_test = split_data(X, y)
