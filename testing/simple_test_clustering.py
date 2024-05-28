import numpy as np
from PIL import Image
from models.KMeans import KMeans


def get_pixels(path):
    image = Image.open(path)
    image = image.convert("RGB")
    pixels = np.array(image.getdata())
    print(f'Number of pixels: {pixels.shape[0]}')
    return pixels


if __name__ == '__main__':
    bg_colors = [
        [[255, 255, 255]],  # hand0
        [[255, 255, 255]],  # hand1
        [[217, 217, 217]],  # hand2
        [[235, 148, 174]],  # hand3
        [[217, 217, 217], [253, 253, 253]],  # hand4
        [[166, 166, 166], [255, 255, 255]],  # hand5
        [[0, 0, 0]]  # hand6
    ]

    for i in range(7):
        print(f'Picture {i}')
        pixels = get_pixels(f'../dataset/hand{i}.png')
        X_train, true_labels = pixels, [0 if list(p) in bg_colors[i] else 1 for p in pixels]
        mean = np.mean(X_train, axis=0)
        std_dev = np.std(X_train, axis=0)
        X_train = (X_train - mean) / std_dev
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X_train)
        class_centers, classification = kmeans.predict(X_train)
        if classification[0] == 1:
            classification = list(map(lambda x: 1 if x == 0 else 0, classification))
        misclustered = np.sum(true_labels != classification)
        print(f'Accuracy: {1 - misclustered / len(pixels)}\n')

    # Picture 0
    # Number of pixels: 698896
    # Accuracy: 0.9950679357157575

    # Picture 1
    # Number of pixels: 317520
    # Accuracy: 0.9999968505920886

    # Picture 2
    # Number of pixels: 526462
    # Accuracy: 0.9999981005276735

    # Picture 3
    # Number of pixels: 784996
    # Accuracy: 0.9939184403487407

    # Picture 4
    # Number of pixels: 318634
    # Accuracy: 0.901165600657808

    # Picture 5
    # Number of pixels: 454528
    # Accuracy: 0.9999977999155167

    # Picture 6
    # Number of pixels: 273529
    # Accuracy: 0.9949657988732457
