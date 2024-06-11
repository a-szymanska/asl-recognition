import cv2
import tensorflow as tf
import os
import numpy as np

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cap = cv2.VideoCapture(0)


model=tf.keras.models.load_model(os.path.join(project_dir,'trained_models/cnn_TFnew_dataBIG.keras'))
#if os.path.isfile(os.path.join(project_dir,'data/new_dataset/ASL_Alphabet_Dataset/asl_alphabet_train/N/N3000.jpg')):
#    frame=cv2.imread(os.path.join(project_dir,'data/new_dataset/ASL_Alphabet_Dataset/asl_alphabet_train/N/N3000.jpg'))
#else:
#    print("debil")
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1)
    if key == ord('c'):
        break


#cv2.imshow('Camera', frame)
height, width, _ = frame.shape
start_x = (width - height) // 2
frame = frame[:, start_x:start_x + 480]

no_pixels=64
frame = cv2.resize(frame, (no_pixels, no_pixels))
frame=np.expand_dims(frame, axis=0)
print(frame.shape)

pred=model.predict(frame)
Y=np.argmax(pred,axis=1)
print(chr(65 + Y[0]))
print(pred)
cap.release()
cv2.destroyAllWindows()