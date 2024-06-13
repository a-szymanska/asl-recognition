from prepare import *
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_file = os.path.join(project_dir, 'data/train3.npz')
test_data_file = os.path.join(project_dir, 'data/test3.npz')
X_train, Y_train = read_data(train_data_file)
X_test, Y_test = read_data(test_data_file)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(26))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(X_train, Y_train,epochs=2)


test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(test_acc)
model.save(os.path.join(project_dir, 'trained_models/cnn_TF.keras'))


