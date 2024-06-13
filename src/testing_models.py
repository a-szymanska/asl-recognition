from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.k_neighbours import KNeighbours
from models.CNN import CNN
from prepare_clustering import *
import os
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_labels = os.path.join(project_dir, 'data/train_clustered.npz')
    test_labels = os.path.join(project_dir, 'data/test_clustered.npz')

    X_train, Y_train = read_data(train_labels)
    X_test, Y_test = read_data(test_labels)

    print('Shape of train dataset:', X_train.shape, Y_train.shape)
    print('Shape of test dataset:', X_test.shape, Y_test.shape)

    # Standardize the data - maybe not necessary?
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cnn = CNN()
    cnn.add_layer(CNN.ConvolutionLayer((3, 3), 32))
    cnn.add_layer(CNN.MaxPoolLayer((2, 2)))
    cnn.add_layer(CNN.DenseLayer(26))
    models = {
        "Decision Tree": DecisionTree(),
        "Random Forest": RandomForest(),
        "K-NN": KNeighbours(),
        "CNN": cnn
    }

    for model_name, model in models.items():
        print(model_name)
        model.fit(X_train_scaled, Y_train)
        Y_pred = model.predict(X_test_scaled)
        model.score(Y_test, Y_pred)
