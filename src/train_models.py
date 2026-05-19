import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_and_save():
    """
    trains enterprise-level models linking patient data to operational data.
    """
    os.makedirs('models', exist_ok=True)
    
    print("loading data...")
    patients_df = pd.read_csv('data/raw/patients.csv')
    appointments_df = pd.read_csv('data/raw/appointments.csv')
    admissions_df = pd.read_csv('data/raw/admissions.csv')
    
    # --- NO-SHOW MODEL ---
    print("training no-show model...")
    ns_full = pd.merge(appointments_df, patients_df, on='patient_id', how='left')
    
    le_dept = LabelEncoder()
    ns_full['department_code'] = le_dept.fit_transform(ns_full['department'])
    
    le_cond = LabelEncoder()
    ns_full['chronic_code'] = le_cond.fit_transform(ns_full['chronic_condition'])
    
    le_cons = LabelEncoder()
    ns_full['consultation_code'] = le_cons.fit_transform(ns_full['consultation_type'])
    
    features_ns = ['department_code', 'chronic_code', 'consultation_code', 
                   'wait_time_hours', 'distance_to_hospital_km', 'historical_no_show_rate', 'age']
                   
    x_ns = ns_full[features_ns]
    y_ns = ns_full['no_show']
    
    x_train_ns, x_test_ns, y_train_ns, y_test_ns = train_test_split(x_ns, y_ns, test_size=0.2, random_state=42)
    no_show_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    no_show_model.fit(x_train_ns, y_train_ns)
    
    print(f"no-show model accuracy: {accuracy_score(y_test_ns, no_show_model.predict(x_test_ns)):.4f}")
    
    # save encoders and model
    joblib.dump(le_dept, 'models/le_dept.pkl')
    joblib.dump(le_cond, 'models/le_cond.pkl')
    joblib.dump(le_cons, 'models/le_cons.pkl')
    joblib.dump(no_show_model, 'models/no_show_rf.pkl')
    
    # --- BED OCCUPANCY MODEL ---
    print("training occupancy model...")
    occ_full = pd.merge(admissions_df, patients_df, on='patient_id', how='left')
    
    le_ward = LabelEncoder()
    occ_full['ward_code'] = le_ward.fit_transform(occ_full['ward'])
    
    occ_full['chronic_code'] = le_cond.transform(occ_full['chronic_condition'])
    
    le_adm = LabelEncoder()
    occ_full['admission_code'] = le_adm.fit_transform(occ_full['admission_type'])
    
    features_occ = ['ward_code', 'chronic_code', 'admission_code', 'severity_level', 'age']
    
    x_occ = occ_full[features_occ]
    y_occ = occ_full['length_of_stay_hours']
    
    x_train_occ, x_test_occ, y_train_occ, y_test_occ = train_test_split(x_occ, y_occ, test_size=0.2, random_state=42)
    occupancy_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    occupancy_model.fit(x_train_occ, y_train_occ)
    
    print(f"occupancy model mse: {mean_squared_error(y_test_occ, occupancy_model.predict(x_test_occ)):.4f}")
    
    joblib.dump(le_ward, 'models/le_ward.pkl')
    joblib.dump(le_adm, 'models/le_adm.pkl')
    joblib.dump(occupancy_model, 'models/occupancy_rf.pkl')
    
    print("models and encoders successfully trained and saved in models/")

if __name__ == "__main__":
    train_and_save()
