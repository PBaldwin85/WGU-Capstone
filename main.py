import os
import random
from tkinter import filedialog
import keras
import numpy as np
from PIL import Image, ImageTk
from keras.src.saving.saving_api import load_model
import cv2
import tkinter as tk
import keras.backend as K


@keras.utils.register_keras_serializable()
def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


new_model = load_model(os.path.join('model', 'catDogClassification.keras'))

root = tk.Tk()
root.title("Pets")
root.geometry("500x500")

browseImages = []
currentIndex = 0
predictionText = ""

for filename in os.listdir("Images/Cats"):
    image_path = os.path.join("Images/Cats", filename)
    image = Image.open(image_path)
    image = np.array(image)
    browseImages.append(image)

for filename in os.listdir("Images/Dogs"):
    image_path = os.path.join("Images/Dogs", filename)
    image = Image.open(image_path)
    image = np.array(image)
    browseImages.append(image)

browseImages = np.array(browseImages)


def showImage(index):
    global currentIndex
    global predictionText
    currentIndex = index
    image_rbg = cv2.cvtColor(browseImages[currentIndex], cv2.COLOR_BGR2RGB)
    photo = Image.fromarray(image_rbg)
    converted_image = ImageTk.PhotoImage(photo)
    label.config(image=converted_image)
    label.image = converted_image
    expanded = np.expand_dims(image_rbg, 0)
    predicted = new_model.predict(expanded)

    if float(predicted) < 0.5:
        predictionText = "Cat"
        label2.config(text=predictionText)
    else:
        predictionText = "Dog"
        label2.config(text=predictionText)


def nextImage():
    global currentIndex
    currentIndex = random.randint(0, len(browseImages) - 1)
    showImage(currentIndex)


def browseFile():
    image_path = filedialog.askopenfilename(filetypes=[('Image', ['.jpg', 'jpeg', 'bmp'])])
    if image_path:
        image = Image.open(image_path)
        image = image.resize((250, 250))
        image = np.array(image)

        image_rbg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = Image.fromarray(image_rbg)
        converted_image = ImageTk.PhotoImage(photo)
        label.config(image=converted_image)
        label.image = converted_image

        expanded = np.expand_dims(image_rbg, 0)
        predicted = new_model.predict(expanded)

        if float(predicted) < 0.5:
            predictionText = "Cat"
            label2.config(text=predictionText)

        else:
            predictionText = "Dog"
            label2.config(text=predictionText)
    else:
        return


label = tk.Label(root, width = 250, height = 250, bg = 'black', fg = 'yellow')
label.pack()

buttonframe = tk.Frame(root)
buttonframe.columnconfigure(0, weight=1)

btn1 = tk.Button(buttonframe, text="Browse Next Image", font=('Arial', 18), command=nextImage)
btn1.grid(row=0, column=0)
buttonframe.pack()

label2 = tk.Label(root, font=("Arial", 14))
label2.pack()

btn1 = tk.Button(buttonframe, text="Upload Image", font=('Arial', 18), command=browseFile)
btn1.grid(row=1, column=0)
buttonframe.pack()

showImage(0)

btn1 = tk.Button(buttonframe, text="Exit", font=('Arial', 18), command=exit)
btn1.grid(row=3, column=0)
buttonframe.pack()

root.mainloop()

