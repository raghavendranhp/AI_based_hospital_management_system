import streamlit as st
import pandas as pd
import os
from src.data_generator import HMSDataGenerator
from src.ml_models import HMSPredictor
from src.llm_agent import InsightGenerator

def render_data_overview():
    """
    renders the data overview section.
    """
    st.header("Data Overview")
    
    #checking if data exists, if not generate it
    if not os.path.exists("data/raw/patients.csv"):
        st.info("generating synthetic hospital data...")
        generator = HMSDataGenerator()
        generator.generate_all()
        st.success("data generation complete.")
        
    try:
        patients_df = pd.read_csv("data/raw/patients.csv")
        appointments_df = pd.read_csv("data/raw/appointments.csv")
        
        st.subheader("Patient Records")
        st.dataframe(patients_df.head(10))
        
        st.subheader("Appointments")
        st.dataframe(appointments_df.head(10))
    except FileNotFoundError:
        st.error("data files not found. please check the data directory.")

def render_predictive_analytics():
    """
    renders the predictive analytics section.
    """
    st.header("Predictive Analytics")
    
    if not os.path.exists("models/no_show_rf.pkl"):
        st.error("pre-trained model not found. please run the training script.")
        return
        
    try:
        appointments_df = pd.read_csv("data/raw/appointments.csv")
        predictor = HMSPredictor()
        
        st.subheader("Simulate No-Show Prediction")
        departments = appointments_df['department'].unique()
        selected_dept = st.selectbox("Select Department", departments)
        
        #dummy encoding for prediction
        dept_code = appointments_df['department'].astype('category').cat.categories.get_loc(selected_dept)
        input_data = pd.DataFrame({'department_code': [dept_code]})
        
        prediction = predictor.predict_no_show(input_data)
        result = "High Risk of No-Show" if prediction[0] == 1 else "Likely to Attend"
        st.info(f"Prediction for {selected_dept}: {result}")
        
    except FileNotFoundError:
        st.error("data not found. please visit data overview first.")

def render_resource_optimization():
    """
    renders the resource optimization section.
    """
    st.header("Resource Optimization")
    
    if not os.path.exists("models/occupancy_rf.pkl"):
        st.error("pre-trained model not found. please run the training script.")
        return
        
    try:
        admissions_df = pd.read_csv("data/raw/admissions.csv")
        predictor = HMSPredictor()
        
        st.subheader("Predict Bed Occupancy Duration")
        wards = admissions_df['ward'].unique()
        selected_ward = st.selectbox("Select Ward", wards)
        
        ward_code = admissions_df['ward'].astype('category').cat.categories.get_loc(selected_ward)
        input_data = pd.DataFrame({'ward_code': [ward_code]})
        
        prediction = predictor.predict_occupancy(input_data)
        st.info(f"Estimated average length of stay in {selected_ward} ward: {prediction[0]:.2f} hours")
        
    except FileNotFoundError:
        st.error("data not found. please visit data overview first.")

def render_ai_insights():
    """
    renders the ai insights section utilizing the groq api.
    """
    st.header("AI Insights")
    
    if st.button("Generate Hospital Insights"):
        try:
            appointments_df = pd.read_csv("data/raw/appointments.csv")
            admissions_df = pd.read_csv("data/raw/admissions.csv")
            
            #preparing a summary
            total_appointments = len(appointments_df)
            no_show_rate = appointments_df['no_show'].mean() * 100
            avg_stay = admissions_df['length_of_stay_hours'].mean()
            
            summary = (
                f"Total Appointments: {total_appointments}\n"
                f"Average No-Show Rate: {no_show_rate:.2f}%\n"
                f"Average Length of Stay: {avg_stay:.2f} hours\n"
            )
            
            with st.spinner("Analyzing hospital data..."):
                agent = InsightGenerator()
                insights = agent.generate_insights(summary)
                
            st.markdown(insights)
            
        except Exception as e:
            st.error(f"error generating insights: {e}")
