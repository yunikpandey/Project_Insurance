import streamlit as st
import pickle
import pandas as pd

# Load the model once when app starts
@st.cache_resource  # This makes it load only once
def load_model():
    with open('insurance_model.pickle', 'rb') as f:
        package = pickle.load(f)
    return package

package = load_model()
scaler = package['scaler']
poly = package['poly_features']
model = package['model']
expected_cols = package['expected_columns']

st.title("Medical Insurance Charges Predictor")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children", 0, 10, 0)
smoker = st.selectbox("Smoker?", ["no", "yes"])
region = st.selectbox("Region", ["northwest", "southeast", "southwest"])

if st.button("Predict Charges"):

    # Encode categorical inputs EXACTLY like training
    sex_val = 1 if sex == "male" else 0
    smoker_val = 1 if smoker == "yes" else 0

    # Base input dictionary
    input_dict = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex': sex_val,
        'smoker': smoker_val,
        'northwest': 0,
        'southeast': 0,
        'southwest': 0
    }

    # One-hot encode region manually (drop_first=True)
    if region == "northwest":
        input_dict['northwest'] = 1
    elif region == "southeast":
        input_dict['southeast'] = 1
    elif region == "southwest":
        input_dict['southwest'] = 1
    # northeast → all zeros

    # Create DataFrame in correct order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[expected_cols]

    # Scale → Polynomial → Predict
    input_scaled = scaler.transform(input_df.values)
    input_poly = poly.transform(input_scaled)
    prediction = model.predict(input_poly)[0]

    st.success(f"Predicted Annual Medical Charges: **${prediction:,.2f}**")
