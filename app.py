import streamlit as st
import pandas as pd
import joblib

# Label mappings (must match your training encoding)
education_map = {
    "HS-grad": 0, "Some-college": 1, "Bachelors": 2,
    "Masters": 3, "PhD": 4, "Assoc": 5
}

occupation_map = {
    "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
    "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6,
    "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
    "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12,
    "Armed-Forces": 13
}

# Load the model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", list(education_map.keys()))
occupation = st.sidebar.selectbox("Job Role", list(occupation_map.keys()))
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Encode input
input_df = pd.DataFrame({
    'age': [age],
    'education': [education_map[education]],
    'occupation': [occupation_map[occupation]],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### 🔎 Encoded Input Data (as seen by the model):")
st.write(input_df)

if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    # Encode batch input
    batch_data['education'] = batch_data['education'].map(education_map)
    batch_data['occupation'] = batch_data['occupation'].map(occupation_map)

    st.write("Uploaded data preview (encoded):", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
