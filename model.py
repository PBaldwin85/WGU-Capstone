import os
import numpy as np
from PIL import Image
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.backend as K

images = []
labels = []

for filename in os.listdir("C:/Users/Pat/Desktop/catdogResize/Cat"):
    image_path = os.path.join("C:/Users/Pat/Desktop/catdogResize/Cat", filename)
    image = Image.open(image_path)
    image = np.array(image)
    images.append(image)
    labels.append(0)

for filename in os.listdir("C:/Users/Pat/Desktop/catdogResize/Dog"):
    image_path = os.path.join("C:/Users/Pat/Desktop/catdogResize/Dog", filename)
    image = Image.open(image_path)
    image = np.array(image)
    images.append(image)
    labels.append(1)

images = np.array(images)
labels = np.array(labels)

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(128, (2, 2), 2, activation='relu', input_shape=(250, 250, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(128, (2, 2), 2, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128, (2, 2), 2, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


model.compile('adam', loss='binary_crossentropy', metrics=['accuracy', get_f1])

history = model.fit(train_images, train_labels, epochs=15, validation_data=(val_images, val_labels))


model.save(os.path.join('model', 'catDogClassification.keras'))

plt.plot(history.history['val_get_f1'])
plt.title('F1 Score')
plt.ylabel('F1Score')
plt.xlabel('Epoch')
plt.legend(['Validation F1Score'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
plt.show()
