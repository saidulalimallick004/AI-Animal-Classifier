import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image 


model=tf.keras.models.load_model('Model/cat&dogClassifier.h5')

st.title("Cat Dog Classifier ")

st.header("Upload An Image to Predict")

uploaded_file=st.file_uploader('Upload An Image',type=['jpg','jpeg','png'])

if st.button("Predict"):
    if uploaded_file is not None:
        img=Image.open(uploaded_file)

        st.image(img,caption="Uploaded Image",use_container_width=True )
        
        img=img.resize((150,150))

        img_array=image.img_to_array(img)/255.0

        img_array=np.expand_dims(img_array,axis=0)

        prediction=model.predict(img_array)

        result="Dog" if prediction[0][0]>0.5 else "Cat"


        st.success(f"Predicted Image : **{result}**")

    else: 
        st.error("Upload an Image")