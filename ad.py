import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved Decision Tree model
@st.cache_resource
def load_model():
    with open('dt.sav', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Define the mapping dictionary for categorical features
label_encoders = {
    'gender': {'Female': 0, 'Male': 1, 'Non-Binary': 2},
    'device_type': {'Desktop': 0, 'Mobile': 1, 'Tablet': 2},
    'ad_position': {'Top': 2, 'Side': 1, 'Bottom': 0},
    'browsing_history': {
        'Shopping': 3, 'Entertainment': 1, 'Education': 0, 'Social Media': 4, 'News': 2
    },
    'time_of_day': {'Afternoon': 0, 'Morning': 2, 'Night': 3, 'Evening': 1}
}

# Function to preprocess input data
def preprocess_input(data):
    for column, encoder in label_encoders.items():
        if column in data.columns:
            data[column] = data[column].replace(encoder)
    return data

# Main app content
def main():
    # Add the image at the top of the app
    st.image("data:image/png;base64,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", 
             caption="Ad Click Prediction App", 
             use_container_width=True)

    # Sidebar for input variables
    st.sidebar.header("Input Features")

    # Input fields in the sidebar
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.sidebar.selectbox("Gender", options=['Female', 'Male', 'Non-Binary'])
    device_type = st.sidebar.selectbox("Device Type", options=['Desktop', 'Mobile', 'Tablet'])
    ad_position = st.sidebar.selectbox("Ad Position", options=['Top', 'Side', 'Bottom'])
    browsing_history = st.sidebar.selectbox("Browsing History", options=[
        'Shopping', 'Entertainment', 'Education', 'Social Media', 'News'
    ])
    time_of_day = st.sidebar.selectbox("Time of Day", options=['Afternoon', 'Morning', 'Night', 'Evening'])

    # Add a submit button in the sidebar
    if st.sidebar.button("Predict"):
        try:
            # Encode categorical features
            input_data = {
                'age': age,
                'gender': gender,
                'device_type': device_type,
                'ad_position': ad_position,
                'browsing_history': browsing_history,
                'time_of_day': time_of_day
            }

            input_df = pd.DataFrame([input_data])
            input_df = preprocess_input(input_df)

            # Make prediction
            prediction = model.predict(input_df)

            # Display the result in the main area
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.markdown("<h3 style='color: green;'>The user is likely to click on the ad.</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: red;'>The user is unlikely to click on the ad.</h3>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()