import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import cv2 as cv2
from tensorflow.keras.models import load_model
import sys
import random
import numpy as np

dir = "D:\VND-pics"
cap = cv2.VideoCapture(0)

# Dinh nghia class
class_name = ['00000',"1000","2000","5000","10000","20000","50000","100000","200000","500000"]

base_model = VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3))
base_model.trainable = False
my_model = tf.keras.Sequential([base_model,tf.keras.layers.Flatten(name="flatten"),tf.keras.layers.Dense(4096, activation='relu', name='fc1'),tf.keras.layers.Dense(4096, activation='relu', name='fc2'),tf.keras.layers.Dense(9,activation='softmax', name='predictions')])
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
my_model = my_model()
my_model.load_weights("weights-19-1.00.hdf5")
# Print out model sum
my_model = load_model('D:\\VND-pics\\model.h5')

while(True):
    # Capture frame-by-frame
    #

    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None,fx=0.5,fy=0.5)
    # Resize
    image = image_org.copy()
    image = cv2.resize(image, dsize=(224, 224))
    image = image.astype('float')*1./255
    # Convert to tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
    print(np.max(predict[0],axis=0))
    if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):


        # Show image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1.5
        color = (0, 255, 0)
        thickness = 2

        cv2.putText(image_org, class_name[np.argmax(predict)], org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()