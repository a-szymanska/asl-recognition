from prepare import *
import sys
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.CNN import CNN


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_file=os.path.join(project_dir,'data/train3.npz')
test_data_file=os.path.join(project_dir,'data/test3.npz')
X_train,Y_train=read_data(train_data_file)
X_test,Y_test=read_data(test_data_file)

model = CNN()
model.add_layer(CNN.ConvolutionLayer((3,3),32))
model.add_layer(CNN.MaxPoolLayer((2,2)))
model.add_layer(CNN.DenseLayer(26))


start_time=time.time()
print(model.fit(X_train,Y_train))
end_time=time.time()
print(end_time-start_time)

