import pickle
import pandas as pd
import numpy as np
import keras
from PIL import Image, ImageOps
import streamlit as st
from streamlit_option_menu import option_menu
from keras.models import load_model
from io import BytesIO, StringIO



#@st.cache(allow_output_mutation=True)
# loading the saved models
#glaucoma_model = pickle.load(open('C:/Users/Admin/Project/R5.sav', 'rb'))
#model = load_model('R50.h5')
#cataract_model = pickle.load(open('C:/Users/Admin/Project/heart_disease_model.sav','rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Eye Disease Detection System',
                          
                          ['Glaucoma Prediction',
                           'Cataract Prediction'],
                          icons=['person','person'],
                          default_index=0)
                        


# Glaucoma Prediction Page
if (selected == 'Glaucoma Prediction'):
    
    # page title
    st.title('GLAUCOMA PREDICTION USING AI')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        NAME = st.text_input('Patients Name')
        
    with col2:
        AGE = st.text_input('Patients Age')

    @st.cache(suppress_st_warning=True)
    def teachable_machine_classification(img, weights_file):
        # Load the model
        
        model = keras.models.load_model(weights_file)
        #model = pickle.load(open(weights_file, 'rb'))

        # Create the array of the right shape to feed into the keras model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = img
        #image sizing
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        return np.argmax(prediction) # return position of the highest probability
    
    uploaded_file = st.file_uploader("Choose a fundus image: ", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Fundus Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'inception_model.h5')
        print(label)
        if label == 0:
            st.write("Patient is Negative")
        else:
            st.write("Patient is Positive")









   
    

  
 
   
        
    
   
    
    
    
    










    # code for Prediction
    # Glaucoma_diagnosis = ''
    
    # # creating a button for Prediction
    
    # if st.button('Glaucoma Test Result'):
    #     glaucoma_prediction = glaucoma_model.predict(img_array)
        
    #     if (glaucoma_prediction[0] == 1):
    #       Glaucoma_diagnosis = 'The person is Glaucomatic'
    #     else:
    #       Glaucoma_diagnosis = 'The person is not Glaucomatic'
        
    # st.success(Glaucoma_diagnosis)

# Cataract Prediction Page
# if (selected == 'Cataract Prediction'):
    
#     # page title
#     st.title('CATARACT PREDICTION USING AI')
    
    
#     # getting the input data from the user
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         NAME = st.text_input('Patients Name')
        
#     with col2:
#         AGE = st.text_input('Patients Age')
    
#     uploaded_file = st.file_uploader("Upload Eye image:")
#     if uploaded_file is not None:

#         # To read file as bytes:
#         bytes_data = uploaded_file.getvalue()
#         st.write(bytes_data)

#         # To convert to a string based IO:
#         stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#         st.write(stringio)

#         # To read file as string:
#         string_data = stringio.read()
#         st.write(string_data)

#         # Can be used wherever a "file-like" object is accepted:
#         dataframe = pd.read_csv(uploaded_file)
#         st.write(dataframe)

        
        
    
   
    
    
    
    
#     # code for Prediction
#     Cataract_diagnosis = ''
    
#     # creating a button for Prediction
    
#     if st.button('Cataract Test Result'):
#         cataract_prediction = diabetes_model.predict([[NAME,AGE,uploaded_file]])
        
#         if (cataract_prediction[0] == 1):
#           Cataract_diagnosis = 'CATARACT POSITIVE'
#         else:
#           Cataract_diagnosis = 'CATARACT NEGATIVE'
        
#     st.success(Cataract_diagnosis)