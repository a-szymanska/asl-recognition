from prepare import *
from prepare_clustering import *
import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_labels = os.path.join(project_dir, 'data/train_clustered.npz')
    test_labels = os.path.join(project_dir, 'data/test_clustered.npz')

    X_train, Y_train = read_data(train_labels)
    X_test, Y_test = read_data(test_labels)

    print('Shape of train dataset:', X_train.shape, Y_train.shape)
    print('Shape of test dataset:', X_test.shape, Y_test.shape)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K-NN": KNeighborsClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    # Train and evaluate models
    results = {}

    for model_name, model in models.items():
        print(model_name)
        model.fit(X_train_scaled, Y_train)
        Y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(Y_test, Y_pred)
        # precision = precision_score(Y_test, Y_pred, average='weighted')
        # recall = recall_score(Y_test, Y_pred, average='weighted')
        # f1 = f1_score(Y_test, Y_pred, average='weighted')
        # conf_matrix = confusion_matrix(Y_test, Y_pred)
        # class_report = classification_report(Y_test, Y_pred)

        print(model_name)
        print("Accuracy", accuracy, '\n')

