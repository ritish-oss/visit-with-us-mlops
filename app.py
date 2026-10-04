import streamlit as st
import pandas as pd
import joblib

st.title("Visit with Us - Wellness Tourism Predictor")

@st.cache_resource
def get_model():
    return joblib.load("models/model.joblib")

model = get_model()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 32)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    passport = st.selectbox("Passport", [1, 0])
    own_car = st.selectbox("Own Car", [1, 0])
    income = st.number_input("Monthly Income", value=22000, step=1000)

with col2:
    trips = st.slider("Trips per year", 1, 15, 3)
    persons = st.slider("Total Accompanying Persons", 1, 6, 2)
    children = st.slider("Children under 5", 0, 4, 0)
    stars = st.selectbox("Preferred Hotel Star", [3, 4, 5])
    contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.slider("Duration of Pitch (min)", 5, 60, 15)
    satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    followups = st.slider("Number of Follow-ups", 1, 6, 3)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

if st.button("Predict Purchase Propensity"):
    row = pd.DataFrame([{
        "Age": age, "TypeofContact": contact, "CityTier": city_tier,
        "Occupation": occupation, "Gender": gender, "NumberOfPersonVisiting": persons,
        "PreferredPropertyStar": stars, "MaritalStatus": marital, "NumberOfTrips": trips,
        "Passport": passport, "OwnCar": own_car, "NumberOfChildrenVisiting": children,
        "Designation": designation, "MonthlyIncome": income,
        "PitchSatisfactionScore": satisfaction, "ProductPitched": pitched,
        "NumberOfFollowups": followups, "DurationOfPitch": duration
    }])

    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]

    st.write("---")
    if pred == 1:
        st.success(f"Likely to Purchase! (Confidence: {prob*100:.1f}%)")
    else:
        st.warning(f"Unlikely to Purchase. (Confidence: {(1-prob)*100:.1f}%)")
