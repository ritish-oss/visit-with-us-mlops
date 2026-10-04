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
    # Read 1 row from travel_package.csv which is committed in the repo
    raw_sample = pd.read_csv("data/travel_package.csv", nrows=1)
    
    # Drop non-feature columns
    features_sample = raw_sample.drop(columns=["CustomerID", "ProdTaken"], errors="ignore")
    input_df = pd.DataFrame(columns=features_sample.columns, index=[0])

    values = {
        "age": age,
        "typeofcontact": type_of_contact,
        "citytier": city_tier,
        "occupation": occupation,
        "gender": gender,
        "numberofpersonvisiting": number_of_person_visiting,
        "preferredpropertystar": preferred_property_star,
        "maritalstatus": marital_status,
        "numberoftrips": number_of_trips,
        "passport": passport,
        "owncar": own_car,
        "numberofchildrenvisiting": number_of_children_visiting,
        "designation": designation,
        "monthlyincome": monthly_income,
        "pitchsatisfactionscore": pitch_satisfaction_score,
        "productpitched": product_pitched,
        "numberoffollowups": number_of_followups,
        "durationofpitch": duration_of_pitch
    }

    for col in input_df.columns:
        clean_key = col.lower().replace("-", "").replace("_", "").replace(" ", "")
        if clean_key in values:
            input_df.at[0, col] = values[clean_key]
        else:
            input_df.at[0, col] = features_sample.at[0, col]

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.write("---")
    if pred == 1:
        st.success(f"### Likely to Purchase! (Confidence: {prob*100:.1f}%)")
    else:
        st.info(f"### Unlikely to Purchase. (Confidence: {(1-prob)*100:.1f}%)")
