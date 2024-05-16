import streamlit as st
from PIL import Image
from PIL import ImageOps
import cv2
import numpy as np
import streamlit.components.v1 as components

import tensorflow as tf
import base64
import io


model=tf.keras.models.load_model('my_model.hdf5')


def predict_age_gender(image):
    #image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
#    img = tf.keras.preprocessing.image.load_img(image,color_mode='grayscale')
    img=image
    img = img.resize((128,128), Image.LANCZOS)
    img = np.array(img)
    img=img/255
    img=img.reshape(1, 128, 128, 1)
    pred = model.predict(img)
    gender_dict = {0:"Male",1:"Female"}

    gender = gender_dict[round(pred[0][0][0])]
    age = round(pred[1][0][0])
    age = max(0, min(100, age))

    return age, gender

st.title("Age and Gender Prediction")

add_selectbox = st.sidebar.selectbox(
    "How do you want to upload the image?",
    ("Please select", "Webcam", "Upload image")
)

if add_selectbox == "Webcam":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        print(img)
        print("image format")
        our_image=ImageOps.grayscale(img)

        if st.button('Predict'):
            age, gender = predict_age_gender(our_image)
            st.success(f'Predicted Age: {age}, Predicted Gender: {gender}')



elif add_selectbox == "Upload image":
    image_file = st.file_uploader("Upload Image", type=['jpeg', 'png', 'jpg', 'webp'])
    if image_file is not None:
        our_image = Image.open(image_file)
  
        st.text("You selected an image from your device.")
        st.image(our_image, width=300)
        our_image=ImageOps.grayscale(our_image)

        if st.button('Predict'):
            age, gender = predict_age_gender(our_image)
            st.success(f'Predicted Age: {age}, Predicted Gender: {gender}')

else:
    st.warning("Please select an option on the sidebar.")