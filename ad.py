import streamlit as st
import pandas as pd
import pickle


filename = 'dt.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Function to predict click probability
def predict_click(features):
  """
  Predicts the click probability based on the given features.

  Args:
    features: A list or array containing the input features.

  Returns:
    The predicted click probability.
  """
  prediction = loaded_model.predict_proba([features])[0][1]
  return prediction

# Streamlit app
st.title("Ad Click Prediction")

# Input fields for features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
device_type = st.selectbox("Device Type", ["Mobile", "Desktop"])
ad_position = st.selectbox("Ad Position", ["Top", "Bottom", "Side"])
browsing_history = st.selectbox("Browsing History", ["Relevant", "Irrelevant"])
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening"])

# Convert categorical features to numerical values
gender_mapping = {"Male": 0, "Female": 1}
device_type_mapping = {"Mobile": 0, "Desktop": 1}
ad_position_mapping = {"Top": 0, "Bottom": 1, "Side": 2}
browsing_history_mapping = {"Relevant": 0, "Irrelevant": 1}
time_of_day_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2}

gender_encoded = gender_mapping[gender]
device_type_encoded = device_type_mapping[device_type]
ad_position_encoded = ad_position_mapping[ad_position]
browsing_history_encoded = browsing_history_mapping[browsing_history]
time_of_day_encoded = time_of_day_mapping[time_of_day]

# Create a list of features
features = [age, gender_encoded, device_type_encoded, ad_position_encoded, browsing_history_encoded, time_of_day_encoded]

# Predict click probability
if st.button("Predict"):
  click_probability = predict_click(features)
  st.write(f"The predicted click probability is: {click_probability:.2f}")
