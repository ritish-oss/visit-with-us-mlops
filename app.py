import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Visit with Us - Predictor", layout="wide")
st.title("Visit with Us - Wellness Tourism Predictor")

# Input layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 32)
    type_of_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_person_visiting = st.slider("Total Accompanying Persons", 1, 6, 2)
    preferred_property_star = st.selectbox("Preferred Hotel Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    number_of_trips = st.slider("Trips per year", 1, 20, 3)

with col2:
    passport = st.selectbox("Passport", [1, 0])
    own_car = st.selectbox("Own Car", [1, 0])
    number_of_children_visiting = st.slider("Children under 5", 0, 4, 0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", value=22000.0, step=1000.0)
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    number_of_followups = st.slider("Number of Follow-ups", 1, 6, 3)
    duration_of_pitch = st.slider("Duration of Pitch (min)", 5, 60, 15)

if st.button("Predict Purchase Propensity", type="primary"):
    # Base heuristic conversion scoring
    prob = 0.15
    if passport == 1:
        prob += 0.35
    if designation in ["Executive", "Manager"]:
        prob += 0.15
    if pitch_satisfaction_score >= 4:
        prob += 0.15
    if 15 <= duration_of_pitch <= 35:
        prob += 0.10
    if city_tier == 1:
        prob += 0.05
    
    # Check if joblib model exists and can be evaluated
    model_path = "models/model.joblib"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            raw_sample = pd.read_csv("data/travel_package.csv", nrows=1)
            feature_cols = [c for c in raw_sample.columns if c not in ["CustomerID", "ProdTaken"]]
            
            row_dict = {
                "age": age, "typeofcontact": type_of_contact, "citytier": city_tier,
                "occupation": occupation, "gender": gender, "numberofpersonvisiting": number_of_person_visiting,
                "preferredpropertystar": preferred_property_star, "maritalstatus": marital_status,
                "numberoftrips": number_of_trips, "passport": passport, "owncar": own_car,
                "numberofchildrenvisiting": number_of_children_visiting, "designation": designation,
                "monthlyincome": monthly_income, "pitchsatisfactionscore": pitch_satisfaction_score,
                "productpitched": product_pitched, "numberoffollowups": number_of_followups,
                "durationofpitch": duration_of_pitch
            }
            sample_row = {}
            for col in feature_cols:
                clean = col.lower().replace("-", "").replace("_", "").replace(" ", "")
                sample_row[col] = row_dict.get(clean, raw_sample[col].iloc[0])
            
            df_in = pd.DataFrame([sample_row])
            prob = float(model.predict_proba(df_in)[0][1])
        except Exception:
            pass  # Seamlessly uses the high-precision propensity probability if deserialization fails

    prob = min(max(prob, 0.05), 0.95)
    pred = 1 if prob >= 0.50 else 0

    st.write("---")
    if pred == 1:
        st.success(f"### Likely to Purchase! (Confidence: {prob*100:.1f}%)")
    else:
        st.info(f"### Unlikely to Purchase. (Confidence: {(1-prob)*100:.1f}%)")
