import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("sign_mnist_train.csv")
labels = df['label']
df.drop(columns = 'label', inplace = True)

##Reshaping data 

reshaped_train = []
for i in range(len(df)):
        reshaped_train.append(df.iloc[i].values.reshape(28,28,1))
        
reshaped_train = np.array(reshaped_train)

train_labels = np.array(labels)
reshaped_labels = OneHotEncoder().fit_transform(train_labels.reshape(-1,1)).toarray()

## Network archtecture 

model = Sequential()
model.add(Conv2D(64, kernel_size = 5, activation = 'relu', input_shape = (28,28, 1), padding = 'same'))

model.add(BatchNormalization())

model.add(Conv2D(16,kernel_size = 3, activation = 'relu'))

model.add(MaxPool2D(4))

model.add(Conv2D(8, kernel_size = 3, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Conv2D(5, kernel_size =3, activation = 'relu'))

model.add(Flatten())

model.add(Dense(24,activation = 'softmax'))

model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

## Model fit 

result = model.fit(reshaped_train, reshaped_labels, validation_split = 0.2, epochs = 200, batch_size = 64, callbacks = [callback])

# Testing 

test = pd.read_csv("sign_mnist_test.csv")
test_labels = np.array(test['label'])
test.drop(columns = 'label', inplace = True)

reshaped_test = []
for i in range(len(test)):
    reshaped_test.append(test.iloc[i].values.reshape(28,28,1))
    
reshaped_test = np.array(reshaped_test)

reshaped_test_labels = OneHotEncoder().fit_transform(test_labels.reshape(-1,1)).toarray()

print("TEST ACCURACY :\n")

print(model.evaluate(reshaped_test, reshaped_test_labels, batch_size = 64))

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(result.history['accuracy'])
ax1.plot(result.history['val_accuracy'])
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper right')
#####
ax2.plot(result.history['loss'])
ax2.plot(result.history['val_loss'])
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper right')
plt.show()