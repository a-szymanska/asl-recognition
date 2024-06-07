import numpy as np
from sklearn.base import BaseEstimator
from sklearn import metrics


class ClassificationModel(BaseEstimator):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    @staticmethod
    def score(y, y_pred):
        y = np.array(y)
        y_pred = np.array(y_pred)

        accuracy = metrics.accuracy_score(y, y_pred)
        precision_macro = metrics.precision_score(y, y_pred, average='macro', zero_division=0)
        precision_micro = metrics.precision_score(y, y_pred, average='micro', zero_division=0)
        precision_weighted = metrics.precision_score(y, y_pred, average='weighted', zero_division=0)
        mcc = metrics.matthews_corrcoef(y, y_pred)  # TODO check if can be used for multiclass problems

        confusion_matrix = metrics.confusion_matrix(y, y_pred)  # May be nicely plotted

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'precision_weighted': precision_weighted,
            'mcc': mcc
        }
