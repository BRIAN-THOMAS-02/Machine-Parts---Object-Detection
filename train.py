import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

tf.config.run_functions_eagerly = True

# Set up TensorFlow session to use available GPU(s)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# initial parameters
epochs = 30
learning_rate = 0.001
batch_size = 32
img_dims = (128, 128, 3)

data = []
labels = []

# load image files from the dataset
image_files = [f for f in
               glob.glob(r'blnw-images-224' + "/**/*", recursive=True) if
               not os.path.isdir(f)]
random.shuffle(image_files)
# print(image_files)

# converting images to arrays and labelling the categories
for img in image_files:

    image = cv2.imread(img)
    #print(image.shape)
    # Check if image is valid
    if image is None or image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] != 3:
        print("Invalid image, skipping...", img)
    else:
        # Resize image
        image = cv2.resize(image, (img_dims[0], img_dims[1]))
        image = img_to_array(image)
        data.append(image)

        label = img.split(os.path.sep)[-2]  # C:\Files\gender_dataset_face\woman\face_1162.jpg
        if label == "bolt":
            label = 0
        elif label == "locatingpin":
            label = 1
        elif label == "nut":
            label = 2
        else:
            label = 3

        labels.append([label])  # [[1], [0], [0], ...]

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print(labels[0])

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

train_X = []
test_X = []
train_y = []
test_y = []

for i in range(len(trainX)):
    # train_X.append(trainX)
    train_X = np.array(train_X)

    # test_X.append(testX)
    test_X = np.array(test_X)

for i in range(len(trainY)):
    # train_y.append(trainY)
    train_y = np.array(train_y)

    # test_y.append(testY)
    test_y = np.array(test_y)

print(trainY[0], trainY[1], trainY[2], trainY[3])
print(testY[0], testY[1], testY[2], testY[3])


trainY = to_categorical(trainY, num_classes=4)  # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=4)
print('trainY : ', trainY[0], trainY[1], trainY[2], trainY[3])
print('testY : ', testY[0], testY[1], testY[2], testY[3])


# augmenting datset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")


# define model
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":  # Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        chanDim = 1

    # The axis that should be normalized, after a Conv2D layer with data_format="channels_first",
    # set axis=1 in BatchNormalization. Dont use BatchNormalization its not at all good, reduces the model accuracy and val_accuracy like anything. Refrain from using Batch Normalization
    

    # First convolutional layer
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=inputShape))
    # model.add(BatchNormalization()) 
    
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.05))

    # Second convolutional layer
    model.add(Conv2D(16, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.05))

    # Third convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.05))

    # Fourth convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.05))

    # Fifth convolutional layer
    # model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.05))

    # Seventh convolutional layer
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.05))

    # Flatten layer
    model.add(Flatten())

    # Dense layers
    model.add(Dense(256, activation='relu'))
    # model.add(BatchNormalization())
    #model.add(Dropout(0.25))

    model.add(Dense(classes, activation='softmax'))

    return model

# build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
              classes=4)

# compile the model
opt = Adam(learning_rate=learning_rate, decay=learning_rate/epochs)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

# train the model
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size, epochs=epochs, verbose=1)

# save the model to disk
model.save('object_detection.model')
#tf.saved_model.save(tf_object_detection, 'tf_object_detection.model', )

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure(dpi=500)
N = epochs
plt.plot(np.arange(0, N), H.history["loss"],         label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"],     label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"],     label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy\n")
plt.xlabel("\nEpoch")
plt.ylabel("Loss/Accuracy\n")
plt.legend(loc="right")

# save plot to disk
plt.savefig('plot_1.png')
