from prepare import *
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data_file = os.path.join(project_dir, 'data/train3.npz')
    test_data_file = os.path.join(project_dir, 'data/test3.npz')

    model = tf.keras.models.load_model(os.path.join(project_dir, 'trained_models/cnn_TF.keras'))
    X_test, Y_test = read_data(test_data_file)

    y_pred = np.argmax(model.predict(X_test),axis=1)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(Y_test, y_pred)
    precision_macro = precision_score(Y_test, y_pred, average='macro')
    precision_micro = precision_score(Y_test, y_pred, average='micro')
    conf_matrix = confusion_matrix(Y_test, y_pred)

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
