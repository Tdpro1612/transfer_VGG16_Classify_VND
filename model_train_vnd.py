#load path file
from google.colab import drive
drive.mount('/content/drive')
path_data = "/content/drive/MyDrive/pix.data"

#load data
import pickle
def load_data():
    file = open(path_data, 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)


    return pixels, labels
X,y = load_data()

#chia data de train,test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=101)

#build model
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

base_model = VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3))
base_model.trainable = False
model = tf.keras.Sequential([base_model,tf.keras.layers.Flatten(name="flatten"),tf.keras.layers.Dense(4096, activation='relu', name='fc1'),tf.keras.layers.Dense(4096, activation='relu', name='fc2'),tf.keras.layers.Dense(10,activation='softmax', name='predictions')])

# Print out model summary
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#tao file checkpoint va lam giau du lieu
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

filepath="weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)
datagen.fit(X_train)

#train data
model_history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
                               epochs=50,# steps_per_epoch=len(X_train)//64,
                               validation_data=datagen.flow(X_test,y_test,
                               batch_size=64),
                               callbacks=callbacks_list)

#save model
model.save("model_vgg16_pretrain.h5")
