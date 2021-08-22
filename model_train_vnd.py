import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

base_model = VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3))
base_model.trainable = False
model = tf.keras.Sequential([base_model,tf.keras.layers.Flatten(name="flatten"),tf.keras.layers.Dense(4096, activation='relu', name='fc1'),tf.keras.layers.Dense(4096, activation='relu', name='fc2'),tf.keras.layers.Dense(9,activation='softmax', name='predictions')])

# Print out model summary
model.summary()