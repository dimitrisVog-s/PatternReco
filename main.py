import imageio
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers
from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import LabelEncoder



'''
with open('Stanford40/ImageSplits/train.txt', 'r') as f:
    train_files = list(map(str.strip, f.readlines()))
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
with open('Stanford40/ImageSplits/test.txt', 'r') as f:
    test_files = list(map(str.strip, f.readlines()))
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))

IMAGE_SIZE = 224
train_images = []
test_images = []

# Resizing all train images to 224x224
for x in train_files:
    img_path = "Stanford40/JPEGImages/" + x
    print(img_path)
    img = imageio.imread(img_path)
    img = img / 255.0
    print("Original Image Size : {}".format(img.shape))
    # Converting 1 channel image into 3 channels
    if len(img.shape) == 2:
        img_temp = cv2.imread(img_path)
        gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        img = np.zeros_like(img_temp)
        img[:, :, 0] = gray
        img[:, :, 1] = gray
        img[:, :, 2] = gray
    new_img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    print("After Image Resize : {}\n".format(new_img.shape))
    train_images.append(new_img)

for y in test_files:
    img_path = "Stanford40/JPEGImages/" + y
    print(img_path)
    img = imageio.imread(img_path)
    img = img / 255.0
    print("Original Image Size : {}".format(img.shape))
    # Converting 1 channel image into 3 channels
    if len(img.shape) == 2:
        img_temp = cv2.imread(img_path)
        gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
        img = np.zeros_like(img_temp)
        img[:, :, 0] = gray
        img[:, :, 1] = gray
        img[:, :, 2] = gray
    new_img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    print("After Image Resize : {}\n".format(new_img.shape))
    test_images.append(new_img)

encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
test_labels = encoder.fit_transform(test_labels)

train_images_split, validation_images_split, train_labels_split, validation_labels_split = train_test_split(train_images, train_labels,
                                                                                    stratify=train_labels, test_size=0.1)

train_images = np.array(train_images_split)
train_labels = np.array(train_labels_split)
validation_images = np.array(validation_images_split)
validation_labels = np.array(validation_labels_split)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), strides=2, kernel_regularizer='l2'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l2'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer='l2'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.BatchNormalization())
model.add(tf.keras.layers.Dense(40, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, horizontal_flip=True)

h = model.fit(x=augmentation.flow(train_images, train_labels), epochs=10, validation_data=(validation_images, validation_labels))

model.save('model.h5')

plt.plot(h.history['accuracy'], label='accuracy')
plt.plot(h.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(h.history['loss'], label='loss')
plt.plot(h.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()

results = model.evaluate(test_images, test_labels, batch_size=128)
print("test loss, test acc:", results)
'''