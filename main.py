import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow import keras 
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
#from keras.preprocessing import image
import keras.utils as image
from sklearn.preprocessing import OneHotEncoder
import cv2
from time import time

import model

def open_camera():
    cam = cv2.VideoCapture(0)

    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

    previous = time()
    delta = 0
    model = load_model("model_weights.h5")

    while(True):
        current = time()
        delta += current - previous
        previous = current
        
        if delta > 5:
            roi = frame[9+1:300-1, 9+1:300-1]
            cv2.imwrite('test.png', roi)
            delta = 0

            img = image.load_img("test.png", target_size=(28, 28))
            img_tensor = image.img_to_array(img)
            #print(img_tensor)
            #img_tensor = img_tensor / 255
            img_tensor = tf.image.rgb_to_grayscale(img_tensor)
            img_tensor = np.expand_dims(img_tensor, axis=0)

            pred = model.predict(img_tensor)
            #print(pred)

            print(alphabet[np.argmax(pred)])

            fig,ax = plt.subplots()
            im = plt.imshow(img)
            ax.set_title(f"Prediction : {alphabet[np.argmax(pred)]}  Confidence : {np.max(pred):.2f}") 
            plt.show()

        
        ret, frame = cam.read()

        cv2.rectangle(frame, (9, 9), (300, 300), (255, 0, 0), 1)

        cv2.imshow("frame", frame)

        """
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("hello")
            roi = frame[9+1:300-1, 9+1:300-1]
            cv2.imwrite('test.png', roi)
        """

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def process_image():
    image = cv2.imread("test.png", 0)
    image = cv2.resize(image, (28, 28))
    return image

def main():
    open_camera()

if __name__ == "__main__":
    main()