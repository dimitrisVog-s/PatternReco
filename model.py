import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import models, layers, callbacks
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import cv2
from keras.models import Sequential, save_model

def create():
    test_sign_data = pd.read_csv('sign_mnist_test.csv').values
    train_sign_data = pd.read_csv('sign_mnist_train.csv').values

    test_labels = test_sign_data[:, 0]
    test_digits = test_sign_data[:, 1:]
    test_digits = np.asarray(test_digits)

    train_labels = train_sign_data[:, 0]
    train_digits = train_sign_data[:, 1:]
    train_digits = np.asarray(train_digits)

    # reshaping train images
    data_array = np.zeros((train_digits.shape[0], 28, 28, 1))
    for i in range(train_digits.shape[0]):
        single_image = train_digits[i,:].reshape(1,-1)
        single_image = single_image.reshape(-1, 28)
        data_array[i,:,:,0] = single_image

    data_array = data_array / 255

    train_digits_split, val_digits_split, train_labels_split, val_labels_split = train_test_split(data_array, train_labels,
                                                                                        stratify=train_labels, test_size=0.1)

    train_digits = np.array(train_digits_split)
    train_labels = np.array(train_labels_split)
    val_digits = np.array(val_digits_split)
    val_labels = np.array(val_labels_split)

    # encoding labels using one hot encoder
    train_labels_df = pd.DataFrame(train_labels)
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(train_labels_df)
    colors_df_encoded = one_hot_encoder.transform(train_labels_df)
    train_labels = pd.DataFrame(data=colors_df_encoded, columns=one_hot_encoder.categories_)

    test_labels_df = pd.DataFrame(test_labels)
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(test_labels_df)
    colors_df_encoded = one_hot_encoder.transform(test_labels_df)
    test_labels = pd.DataFrame(data=colors_df_encoded, columns=one_hot_encoder.categories_)

    val_labels_df = pd.DataFrame(val_labels)
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(val_labels_df)
    colors_df_encoded = one_hot_encoder.transform(val_labels_df)
    val_labels = pd.DataFrame(data=colors_df_encoded, columns=one_hot_encoder.categories_)


    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=5, # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    # building CNN network model
    model = models.Sequential()

    model.add(layers.Conv2D(64,kernel_size=(3,3),activation='swish',input_shape=(28,28,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64,kernel_size=(3,3),activation='swish',input_shape=(28,28,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(32,kernel_size=(3,3),activation='swish'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(tf.keras.layers.Dense(24, activation='softmax'))

    print(model.summary())

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    h = model.fit(train_digits, train_labels, epochs=10, validation_data=(val_digits, val_labels), batch_size=128, callbacks=[early_stopping])

    save_model(model, "test.h5", save_format="h5")

    print(train_digits[0])
    print("-----------------------------------------------------------")

    # reshaping test images
    temp_array = np.zeros((test_digits.shape[0], 28, 28, 1))
    for i in range(test_digits.shape[0]):
        single_image = test_digits[i,:].reshape(1,-1)
        single_image = single_image.reshape(-1, 28)
        temp_array[i,:,:,0] = single_image
    test_digits = temp_array / 255

    results = model.evaluate(test_digits, test_labels, batch_size=128)
    print(results)

    img = image.load_img("test.png", target_size=(28, 28))
    img_tensor = image.img_to_array(img)
    img_tensor = img_tensor / 255
    img_tensor = tf.image.rgb_to_grayscale(img_tensor)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    pred = model.predict(img_tensor)
    print(pred)
        
        