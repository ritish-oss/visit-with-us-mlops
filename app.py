import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Visit with Us - Predictor", layout="wide")
st.title("Visit with Us - Wellness Tourism Predictor")

@st.cache_resource
def get_model():
    return joblib.load("models/model.joblib")

model = get_model()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 32)
    type_of_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.slider("Total Accompanying Persons", 1, 6, 2)
    preferred_property_star = st.selectbox("Preferred Hotel Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    number_of_trips = st.slider("Trips per year", 1.0, 20.0, 3.0)

with col2:
    passport = st.selectbox("Passport", [1, 0])
    own_car = st.selectbox("Own Car", [1, 0])
    number_of_children_visiting = st.slider("Children under 5", 0.0, 4.0, 0.0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", value=22000.0, step=1000.0)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    number_of_followups = st.slider("Number of Follow-ups", 1.0, 6.0, 3.0)
    duration_of_pitch = st.slider("Duration of Pitch (min)", 5.0, 60.0, 15.0)

if st.button("Predict Purchase Propensity", type="primary"):
    data_dict = {
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "ProductPitched": [product_pitched],
        "NumberOfFollowups": [number_of_followups],
        "DurationOfPitch": [duration_of_pitch]
    }
    
    input_df = pd.DataFrame(data_dict)

    # Reorder columns to match the trained model's feature names exactly
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[model.feature_names_in_]

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.write("---")
    if pred == 1:
        st.success(f"### Likely to Purchase! (Confidence: {prob*100:.1f}%)")
    else:
        st.info(f"### Unlikely to Purchase. (Confidence: {(1-prob)*100:.1f}%)")
