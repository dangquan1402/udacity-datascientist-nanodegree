import os
import time

import numpy as np
import requests
import streamlit as st
from PIL import Image

if not os.path.exists("uploaded_images"):
    os.makedirs("uploaded_images")
# Designing the interface
st.title("Male Female Classification App")
# For newline
st.write("\n")

# image = Image.open("sample_image.jpg")
# show = st.image(image, use_column_width=True)

st.sidebar.title("Upload Image")

# Disabling warning
st.set_option("deprecation.showfileUploaderEncoding", False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    u_img = Image.open(uploaded_file)
    # save for later
    u_img.save(os.path.join("uploaded_images", uploaded_file.name))
    st.image(u_img, use_column_width="auto")
    image = np.asarray(u_img) / 255


# For newline
st.sidebar.write("\n")

if st.sidebar.button("Click Here to Classify"):

    if uploaded_file is None:

        st.sidebar.write("Please upload an Image to Classify")

    else:

        with st.spinner("Classifying ..."):
            url = "http://0.0.0.0:5000/predict"  # localhost and the defined port + endpoint
            body = {"image_path": os.path.join("uploaded_images", uploaded_file.name)}
            response = requests.post(url, data=body)
            prediction = response.json()["Prediction"]
            print(prediction)
            time.sleep(2)
            st.success("Done!")

        st.sidebar.header("Model Predicts: ")

        # Classify cat being present in the picture if prediction > 0.5

        if prediction["female"] > 0.5:

            st.sidebar.write("This is a 'Female' picture.", "\n")

            st.sidebar.write("**Probability: **", prediction["female"], "%")

        else:
            st.sidebar.write(" This is a 'Male' picture ", "\n")

            st.sidebar.write("**Probability: **", prediction["male"], "%")
