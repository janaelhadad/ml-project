import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('random_forest_diabetes_model.pkl')

model = load_model()

# Main title
st.title('Diabetes Risk Prediction')
st.markdown("""
This application uses a machine learning model to predict the risk of diabetes based on various health metrics.
Enter your health information below to get a prediction.
""")

# Create columns for better layout
col1, col2 = st.columns(2)

# User input form
with col1:
    st.subheader('Enter Your Health Information')
    
    # Personal information
    gender = st.selectbox('Gender', ['Female', 'Male'])
    age = st.slider('Age', 18, 100, 40)
    
    # Health metrics
    bmi = st.number_input('BMI (kg/m²)', 10.0, 50.0, 25.0, 0.1)
    hba1c = st.number_input('HbA1c Level (%)', 3.0, 15.0, 5.7, 0.1)
    blood_glucose = st.number_input('Blood Glucose Level (mg/dL)', 70, 300, 100, 1)
    
    # Health conditions
    hypertension = st.radio('Do you have Hypertension?', ['No', 'Yes'], horizontal=True)
    heart_disease = st.radio('Do you have Heart Disease?', ['No', 'Yes'], horizontal=True)
    
    # Smoking status
    smoking = st.selectbox('Smoking History', ['non-smoker', 'current', 'past_smoker'])
    
    # Convert inputs to model format
    hypertension_val = 1 if hypertension == 'Yes' else 0
    heart_disease_val = 1 if heart_disease == 'Yes' else 0

# Display input summary on the right
with col2:
    st.subheader('Health Summary')
    
    # Create a summary dataframe
    summary_data = {
        'Metric': ['Gender', 'Age', 'BMI', 'HbA1c Level', 'Blood Glucose Level', 
                  'Hypertension', 'Heart Disease', 'Smoking Status'],
        'Value': [gender, age, f"{bmi:.1f} kg/m²", f"{hba1c:.1f}%", f"{blood_glucose} mg/dL", 
                 hypertension, heart_disease, smoking]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Display summary table
    st.table(summary_df)
    
    # Show information about normal ranges
    st.info("""
    **Normal ranges for reference:**
    - BMI: 18.5-24.9 kg/m²
    - HbA1c: Below 5.7%
    - Blood Glucose (fasting): 70-99 mg/dL
    """)

# Predict button
if st.button('Predict Diabetes Risk', use_container_width=True):
    # Create input data frame for prediction
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension_val],
        'heart_disease': [heart_disease_val],
        'smoking_history': [smoking],
        'bmi': [bmi],
        'HbA1c_level': [hba1c],
        'blood_glucose_level': [blood_glucose]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0, 1] * 100
    
    # Display results
    st.header('Prediction Results')
    
    # Create columns for results display
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        # Display prediction
        if prediction == 1:
            st.error(f"##### Prediction: **Positive Risk for Diabetes**")
        else:
            st.success(f"##### Prediction: **Negative Risk for Diabetes**")
        
        # Display probability
        st.metric(label="Risk Percentage", value=f"{probability:.1f}%")
        
    with res_col2:
        # Create a gauge chart to visualize the risk
        fig, ax = plt.subplots(figsize=(4, 0.3))
        ax.barh([0], [100], color='lightgray', height=0.2)
        ax.barh([0], [probability], color='red' if probability > 50 else 'green', height=0.2)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.tick_top()
        st.pyplot(fig)
        
        # Risk level interpretation
        if probability < 25:
            risk_level = "Low Risk"
            color = "green"
        elif probability < 50:
            risk_level = "Moderate Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"
        
        st.markdown(f"<h5 style='color:{color}'>Risk Level: {risk_level}</h5>", unsafe_allow_html=True)
    
    # Display factors that influence the prediction
    st.subheader('Key Factors Influencing Your Risk')
    
    # Identify high-risk factors
    high_risk_factors = []
    
    if bmi > 30:
        high_risk_factors.append("Your BMI is in the obese range (>30)")
    if hba1c >= 5.7:
        high_risk_factors.append("Your HbA1c level is elevated (≥5.7%)")
    if blood_glucose >= 126:
        high_risk_factors.append("Your fasting blood glucose is elevated (≥126 mg/dL)")
    if hypertension == "Yes":
        high_risk_factors.append("You have hypertension")
    if heart_disease == "Yes":
        high_risk_factors.append("You have heart disease")
    if age > 45:
        high_risk_factors.append("Age above 45 increases diabetes risk")
    if smoking == "current":
        high_risk_factors.append("Current smoking status increases health risks")
    
    # Display risk factors
    if high_risk_factors:
        for factor in high_risk_factors:
            st.warning(factor)
    else:
        st.success("No major high-risk factors identified")
    
    # Recommendations
    st.subheader("Recommendations")
    st.markdown("""
    - **Consult with a healthcare provider** for proper diagnosis and personalized advice
    - Maintain a balanced diet rich in vegetables, lean proteins, and whole grains
    - Regular physical activity (aim for at least 150 minutes per week)
    - Maintain a healthy body weight
    - Get regular health check-ups to monitor your risk factors
    """)
    
    # Disclaimer
    st.caption("""
    **Disclaimer:** This prediction tool is for informational purposes only and is not a substitute 
    for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
    or other qualified health provider with any questions you may have regarding a medical condition.
    """)