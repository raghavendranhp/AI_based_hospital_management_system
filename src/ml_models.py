import pandas as pd
import os
import joblib

class HMSPredictor:
    """
    enterprise machine learning predictor for hospital management tasks.
    """

    def __init__(self, models_dir: str = 'models/'):
        """
        initializes the models and label encoders. loads from disk.
        """
        self.models_dir = models_dir
        
        #load models
        self.no_show_model = joblib.load(os.path.join(models_dir, 'no_show_rf.pkl'))
        self.occupancy_model = joblib.load(os.path.join(models_dir, 'occupancy_rf.pkl'))
        
        #load encoders
        self.le_dept = joblib.load(os.path.join(models_dir, 'le_dept.pkl'))
        self.le_cond = joblib.load(os.path.join(models_dir, 'le_cond.pkl'))
        self.le_cons = joblib.load(os.path.join(models_dir, 'le_cons.pkl'))
        self.le_ward = joblib.load(os.path.join(models_dir, 'le_ward.pkl'))
        self.le_adm = joblib.load(os.path.join(models_dir, 'le_adm.pkl'))
        
    def predict_no_show(self, input_data: dict):
        """
        predicts no-show probability for dynamic patient input data.
        """
        df = pd.DataFrame([input_data])
        df['department_code'] = self.le_dept.transform(df['department'])
        df['chronic_code'] = self.le_cond.transform(df['chronic_condition'])
        df['consultation_code'] = self.le_cons.transform(df['consultation_type'])
        
        features = ['department_code', 'chronic_code', 'consultation_code', 
                   'wait_time_hours', 'distance_to_hospital_km', 'historical_no_show_rate', 'age']
        
        probs = self.no_show_model.predict_proba(df[features])[0]
        return probs[1]

    def predict_occupancy(self, input_data: dict):
        """
        predicts length of stay based on patient severity and demographics.
        """
        df = pd.DataFrame([input_data])
        df['ward_code'] = self.le_ward.transform(df['ward'])
        df['chronic_code'] = self.le_cond.transform(df['chronic_condition'])
        df['admission_code'] = self.le_adm.transform(df['admission_type'])
        
        features = ['ward_code', 'chronic_code', 'admission_code', 'severity_level', 'age']
        
        return self.occupancy_model.predict(df[features])[0]
