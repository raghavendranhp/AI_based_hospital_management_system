import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class HMSPredictor:
    """
    machine learning predictor for hospital management tasks.
    """

    def __init__(self, models_dir: str = 'models/'):
        """
        initializes the models. loads from disk if available.
        """
        self.models_dir = models_dir
        self.no_show_model_path = os.path.join(models_dir, 'no_show_rf.pkl')
        self.occupancy_model_path = os.path.join(models_dir, 'occupancy_rf.pkl')
        
        if os.path.exists(self.no_show_model_path):
            self.no_show_model = joblib.load(self.no_show_model_path)
        else:
            self.no_show_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        if os.path.exists(self.occupancy_model_path):
            self.occupancy_model = joblib.load(self.occupancy_model_path)
        else:
            self.occupancy_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def predict_no_show(self, input_features: pd.DataFrame):
        """
        predicts no-show probability for given appointments.
        """
        return self.no_show_model.predict(input_features)

    def predict_occupancy(self, input_features: pd.DataFrame):
        """
        predicts occupancy lengths.
        """
        return self.occupancy_model.predict(input_features)
