from prepare import *
from models.random_forest import RandomForest
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_labels = os.path.join(project_dir, 'data/train_clustered.npz')
    X_train, Y_train = read_data(train_labels)

    X_train_scaled = StandardScaler().fit_transform(X_train)

    model = RandomForest()
    model.fit(X_train_scaled, Y_train)
    joblib.dump(model, 'trained_models/random_forest.pkl')

    model = joblib.load('trained_models/random_forest.pkl')
    test_labels = os.path.join(project_dir, 'data/test_clustered.npz')
    X_test, Y_test = read_data(test_labels)

    X_test_scaled = StandardScaler().fit_transform(X_test)
    Y_pred = model.predict(X_test_scaled)
    print(accuracy_score(Y_test, Y_pred))
