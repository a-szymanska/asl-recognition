import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


class ClassificationModel(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    @staticmethod
    def score(y, y_pred):
        y = np.array(y)
        y_pred = np.array(y_pred)

        accuracy = accuracy_score(y, y_pred)
        precision_macro = precision_score(y, y_pred, average='macro')
        precision_micro = precision_score(y, y_pred, average='micro')
        conf_matrix = confusion_matrix(y, y_pred)

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
