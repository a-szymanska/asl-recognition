from prepare import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_file=os.path.join(project_dir,'data/train3.npz')
test_data_file=os.path.join(project_dir,'data/test3.npz')

model=tf.keras.models.load_model(os.path.join(project_dir,'trained_models/cnn_TF.keras'))
X_test,Y_test=read_data(test_data_file)

prediction=model.predict(np.array([X_test[0]]))
print(np.argmax(prediction,axis=1))
show_image(X_test[0],Y_test[0])
