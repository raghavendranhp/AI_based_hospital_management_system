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
    st.header("Enterprise Data Overview")
    
    #checking if data exists, if not generate it
    if not os.path.exists("data/raw/patients.csv"):
        st.info("generating synthetic hospital data...")
        generator = HMSDataGenerator()
        generator.generate_all()
        st.success("data generation complete.")
        
    try:
        patients_df = pd.read_csv("data/raw/patients.csv")
        appointments_df = pd.read_csv("data/raw/appointments.csv")
        resources_df = pd.read_csv("data/raw/hospital_resources.csv")
        
        st.subheader("Hospital Resources (Bed Capacity)")
        st.dataframe(resources_df)

        st.subheader("Patient Health Records")
        st.dataframe(patients_df.head(10))
        
        st.subheader("Dynamic Appointments")
        st.dataframe(appointments_df.head(10))
    except FileNotFoundError:
        st.error("data files not found. please check the data directory.")

def render_predictive_analytics():
    """
    renders the predictive analytics section with dynamic patient features.
    """
    st.header("Dynamic No-Show Prediction")
    
    if not os.path.exists("models/no_show_rf.pkl"):
        st.error("pre-trained model not found. please run the training script.")
        return
        
    predictor = HMSPredictor()
    
    st.subheader("Patient & Appointment Details")
    col1, col2 = st.columns(2)
    
    with col1:
        department = st.selectbox("Department", predictor.le_dept.classes_)
        chronic_condition = st.selectbox("Chronic Condition", predictor.le_cond.classes_)
        consultation_type = st.selectbox("Consultation Type", predictor.le_cons.classes_)
        age = st.slider("Patient Age", 1, 100, 45)
        
    with col2:
        distance = st.slider("Distance to Hospital (km)", 1.0, 100.0, 15.0)
        wait_time = st.slider("Wait Time (hours)", 1, 720, 48)
        hist_no_show = st.slider("Historical No-Show Rate", 0.0, 1.0, 0.1)
        
    input_data = {
        'department': department,
        'chronic_condition': chronic_condition,
        'consultation_type': consultation_type,
        'age': age,
        'distance_to_hospital_km': distance,
        'wait_time_hours': wait_time,
        'historical_no_show_rate': hist_no_show
    }
    
    prob = predictor.predict_no_show(input_data)
    
    st.subheader("Prediction Result")
    st.progress(prob)
    st.info(f"Probability of No-Show: {prob * 100:.1f}%")
    
    if prob > 0.6:
        st.warning("High risk of missed appointment. Consider sending a proactive reminder.")
    else:
        st.success("Patient is highly likely to attend.")

def render_resource_optimization():
    """
    renders the resource optimization section tracking active capacity.
    """
    st.header("Resource Optimization & Capacity Tracking")
    
    if not os.path.exists("models/occupancy_rf.pkl"):
        st.error("pre-trained model not found. please run the training script.")
        return
        
    try:
        resources_df = pd.read_csv("data/raw/hospital_resources.csv")
        admissions_df = pd.read_csv("data/raw/admissions.csv")
        
        st.subheader("Current Ward Utilization")
        #simulate active patients as 30% of total admissions for visual purpose
        active_admissions = admissions_df.sample(frac=0.3, random_state=42)
        active_counts = active_admissions['ward'].value_counts().reset_index()
        active_counts.columns = ['ward', 'active_patients']
        
        utilization = pd.merge(resources_df, active_counts, on='ward', how='left').fillna(0)
        utilization['utilization_pct'] = (utilization['active_patients'] / utilization['total_beds']) * 100
        
        for index, row in utilization.iterrows():
            st.write(f"**{row['ward'].capitalize()} Ward** ({int(row['active_patients'])}/{int(row['total_beds'])} beds)")
            progress_val = min(row['utilization_pct'] / 100.0, 1.0)
            st.progress(progress_val)
            if row['utilization_pct'] > 100:
                st.error(f"Overcapacity Warning: {row['ward'].capitalize()} Ward is over capacity!")
            
        st.markdown("---")
        st.subheader("Predict Patient Length of Stay")
        
        predictor = HMSPredictor()
        
        col1, col2 = st.columns(2)
        with col1:
            ward = st.selectbox("Target Ward", predictor.le_ward.classes_)
            chronic_condition = st.selectbox("Patient Condition", predictor.le_cond.classes_)
            admission_type = st.selectbox("Admission Type", predictor.le_adm.classes_)
        with col2:
            severity = st.slider("Severity Level (1-5)", 1, 5, 3)
            age = st.slider("Patient Age ", 1, 100, 60)
            
        input_data = {
            'ward': ward,
            'chronic_condition': chronic_condition,
            'admission_type': admission_type,
            'severity_level': severity,
            'age': age
        }
        
        los = predictor.predict_occupancy(input_data)
        st.info(f"Estimated Length of Stay: {los:.1f} hours ({los/24:.1f} days)")
        
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
