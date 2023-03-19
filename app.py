import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import os
import time
import pandas as pd
import plotly.graph_objs as go


# Define the class labels for the 4 objects
class_labels = ['Nut', 'Washer', 'Locatingpin', 'Bolt']

# Load the trained CNN model
model = tf.keras.models.load_model('object_detection.model2')


# Define a function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    return img
    
    
# Define the Streamlit app
def app():
    # Add a title and description
    st.write("<p style='font-size: 75px;'> Object Classification App </p>", unsafe_allow_html=True)
    st.write('This app classifies an input image into one of the four object classes using a custom pre-trained CNN model.')
    
    st.write("\n\n")
 
    # Define your block of text
    text_block = '''
    This project has been created with the intention of classifying objects used in the mechanial industry in 4 variant classes namely : <b><i><u> Bolt, Locating Pin, Nut, Washer. </b></i></u> <br>
    We have used Convolutional Neural Networks here for training our model which will predict and classify these objects in it's respective classes. Please refer the CNN architecture given below where you can understand the number of layers and neurons and trainable parameters. <br>
    You can also refer the graph below where the train_acc, train_loss, vall_acc and val_loss through the epochs can be understood. <br>
    The Dataset - https://www.kaggle.com/datasets/manikantanrnair/images-of-mechanical-parts-boltnut-washerpin <br>
    Kindly use this dataset, as the model is trained on the same! The model might not perform very well on images taken from google, as it is not trained on the same as of March 2023!
    '''

    # Use st.markdown to style your block
    st.markdown(
            f"""
            <div style='border: 2.5px solid orange; border-radius: 8px; padding: 10px; background-color: #0000'>
            <p style='font-size:30px; font-weight:bold; color: white; margin-bottom: 2.5px'> Explanation : </p>
            <p style='font-size:17px; line-height: 2.5; margin-bottom: 0px'> {text_block} </p>
            </div>
            """,
            unsafe_allow_html=True
        )
       
    
    st.write("")
    st.write("")
    st.write("")
    
    st.write("<p style='font-size: 24px;'> The &nbsp; 4 &nbsp; Objects are : </p>", unsafe_allow_html=True)
    
    # Creating Dataframe for all our classes
    df = pd.DataFrame({'Classes': ['Bolt', 'Locating Pin', 'Nut', 'Washer']})
    
    # Create a string representation of the DataFrame using to_html()
    df_html = df.to_html(classes='centered')

    # Add HTML tags to increase the font size
    html = "<div style='text-align: left;'><p style='font-size: 16px;'> {} </p></div>".format(df_html)

    # Render the HTML using st.markdown()
    st.write(html, unsafe_allow_html=True)
    
    st.write("")
    st.write("")

    with st.expander("Click to see Architecutre of Model"):
        st.subheader('Architecture of CNN Model')
        st.text(
            f"""
            Model: "Sequential"
            ______________________________________________________________________
                 Layer (type)                       Output Shape         Param #   
            ======================================================================
             conv2d_1 (Conv2D)                  (None, 126, 126, 8)       224       
                                                                             
             max_pooling2d_1 (MaxPooling2D)     (None, 63, 63, 8)         0         
                                                                            
             conv2d_2 (Conv2D)                  (None, 61, 61, 16)        1168      
                                                                             
             max_pooling2d_2 (MaxPooling2D)     (None, 30, 30, 16)        0         
                                                                          
             conv2d_3 (Conv2D)                  (None, 28, 28, 32)        4640      
                                                                             
             max_pooling2d_3 (MaxPooling2D)     (None, 14, 14, 32)        0         
                                                                          
             conv2d_4 (Conv2D)                  (None, 12, 12, 16)        4624      
                                                                             
             max_pooling2d_4 (MaxPooling2D)     (None, 6, 6, 16)          0         
                                                                         
             flatten_1 (Flatten)                (None, 576)               0         
                                                                             
             dense_1 (Dense)                    (None, 256)               147712    
                                                                             
             dense_2 (Dense)                    (None, 4)                 1028      
                                                                             
            ======================================================================
            Total params: 159,396
            Trainable params: 159,396
            Non-trainable params: 0
            ______________________________________________________________________
            """
        )
        
    st.write("")
    st.write("")
    
    with st.expander("Click to see Graph"):
        st.subheader('Graph of Object Classification')
        sample_img = cv2.imread('plot_12.png')
        FRAME_WINDOW = st.image(sample_img, channels='BGR')
        st.write('This is the Graph of Performance Metrics throughout the training of our custom Object Classification Model')
        
    with st.sidebar:
        st.write("<p style='font-size: 35px;'> Upload Image </p>", unsafe_allow_html=True)      
        uploaded_file = st.file_uploader('Upload image file in jpeg, png format', type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            with st.spinner("Uploading Image..."):
                time.sleep(3)
                st.success("Done!")
    
    

    # Make a prediction if an image is uploaded
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
      
        # Show the input image and predicted class label
        st.subheader('Input Image')
        
        FRAME_WINDOW1 = st.image(img, channels='BGR')
        FRAME_WINDOW1.image(img, channels='BGR', width=250)

        # Preprocess the input image
        img_array = preprocess_image(img)
        
        # Make a prediction using the trained model
        pred = model.predict(img_array)
        
        
        st.write("<p style='font-size: 25px;'> Predictions </p>", unsafe_allow_html=True)
        st.write(pred)
        #39FF14
        if pred[0][0] > 0.5:
            st.write("<p style='font-size: 50px; color: cyan;'> Bolt : {}%  </p>".format(pred[0][0]*100), unsafe_allow_html=True)
        if pred[0][1] > 0.5:
            st.write("<p style='font-size: 50px; color: #39ff14;'> Locating Pin : {}%  </p>".format(pred[0][1]*100), unsafe_allow_html=True)
        if pred[0][2] > 0.5:
            st.write("<p style='font-size: 50px; color: #DB3EB1;'> Nut : {}%  </p>".format(pred[0][2]*100), unsafe_allow_html=True)
        if pred[0][3] > 0.5:
            st.write("<p style='font-size: 50px; color: #FFC42E;'> Washer : {}%  </p>".format(pred[0][3]*100), unsafe_allow_html=True)

# Run the app
app()
