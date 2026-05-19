import pandas as pd
import os
import joblib
from src.ml_models import HMSPredictor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_save():
    """
    trains models and saves them to the models directory.
    """
    os.makedirs('models', exist_ok=True)
    predictor = HMSPredictor()
    
    print("loading data...")
    appointments_df = pd.read_csv('data/raw/appointments.csv')
    admissions_df = pd.read_csv('data/raw/admissions.csv')
    
    print("training no-show model...")
    appointments_df['department_code'] = appointments_df['department'].astype('category').cat.codes
    x_no_show = appointments_df[['department_code']]
    y_no_show = appointments_df['no_show']
    
    x_train_ns, x_test_ns, y_train_ns, y_test_ns = train_test_split(x_no_show, y_no_show, test_size=0.2, random_state=42)
    predictor.no_show_model.fit(x_train_ns, y_train_ns)
    preds_ns = predictor.no_show_model.predict(x_test_ns)
    print(f"no-show model accuracy: {accuracy_score(y_test_ns, preds_ns):.4f}")
    joblib.dump(predictor.no_show_model, predictor.no_show_model_path)
    
    print("training occupancy model...")
    admissions_df['ward_code'] = admissions_df['ward'].astype('category').cat.codes
    x_occ = admissions_df[['ward_code']]
    y_occ = admissions_df['length_of_stay_hours']
    
    x_train_occ, x_test_occ, y_train_occ, y_test_occ = train_test_split(x_occ, y_occ, test_size=0.2, random_state=42)
    predictor.occupancy_model.fit(x_train_occ, y_train_occ)
    preds_occ = predictor.occupancy_model.predict(x_test_occ)
    print(f"occupancy model mse: {mean_squared_error(y_test_occ, preds_occ):.4f}")
    joblib.dump(predictor.occupancy_model, predictor.occupancy_model_path)
    
    print("models successfully trained and saved in models/")

if __name__ == "__main__":
    train_and_save()
