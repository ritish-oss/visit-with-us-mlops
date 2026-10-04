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
    # Read columns directly from the base dataset
    raw_df = pd.read_csv("data/travel_package.csv", nrows=2)
    feature_cols = [c for c in raw_df.columns if c not in ["CustomerID", "ProdTaken"]]

    mapping = {
        "age": age,
        "typeofcontact": str(type_of_contact),
        "citytier": int(city_tier),
        "occupation": str(occupation),
        "gender": str(gender),
        "numberofpersonvisiting": int(number_of_person_visiting),
        "preferredpropertystar": float(preferred_property_star),
        "maritalstatus": str(marital_status),
        "numberoftrips": float(number_of_trips),
        "passport": int(passport),
        "owncar": int(own_car),
        "numberofchildrenvisiting": float(number_of_children_visiting),
        "designation": str(designation),
        "monthlyincome": float(monthly_income),
        "pitchsatisfactionscore": int(pitch_satisfaction_score),
        "productpitched": str(product_pitched),
        "numberoffollowups": float(number_of_followups),
        "durationofpitch": float(duration_of_pitch)
    }

    row_data = {}
    for col in feature_cols:
        clean_key = col.lower().replace("-", "").replace("_", "").replace(" ", "")
        row_data[col] = mapping.get(clean_key, raw_df[col].iloc[0])

    input_df = pd.DataFrame([row_data])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.write("---")
    if pred == 1:
        st.success(f"### Likely to Purchase! (Confidence: {prob*100:.1f}%)")
    else:
        st.info(f"### Unlikely to Purchase. (Confidence: {(1-prob)*100:.1f}%)")
