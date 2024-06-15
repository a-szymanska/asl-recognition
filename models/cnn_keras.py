from models.model import *
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


class CNNkeras(ClassificationModel):
    def __init__(self):
        super().__init__()
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(26))
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def fit(self, X, y, epochs):
        self.history = self.model.fit(X, y, epochs=epochs)

        plt.plot(self.history.history['accuracy'])
        plt.xlabel('Epoka')
        plt.ylabel('Dokładność')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

    def predict(self, X, y, model_path=None):
        loss, acc = self.model.evaluate(X, y, verbose=2)
        print(f'Accuracy: {acc}')
        if model_path is not None:
            self.model.save(model_path)

        predictions = self.model.predict(X)
        self.y_test = np.array(y)
        self.y_pred = np.argmax(predictions, axis=1)

    def evaluate(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision_macro = precision_score(self.y_test, self.y_pred, average='macro')
        precision_micro = precision_score(self.y_test, self.y_pred, average='micro')
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)

        print("Accuracy", accuracy)
        print("Macro precision", precision_macro)
        print("Micro precision", precision_micro, '\n')

        plt.figure(figsize=(10, 7))
        letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=letters, yticklabels=letters)
        plt.xlabel('Prediction')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.show()
