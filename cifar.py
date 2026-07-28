import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

st.title("CIFAR-10 Image Classifier")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)



loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

st.write("Upload an image and predict its class.")

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((32,32))

    img_array = np.array(img)

   
    img_array = img_array / 255.0

    
    img_array = np.expand_dims(img_array, axis=0)

    
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.subheader("Prediction")

    st.success(f"Predicted Class: **{class_names[predicted_index]}**")

    st.write(f"Confidence: **{confidence:.2f}%**")
