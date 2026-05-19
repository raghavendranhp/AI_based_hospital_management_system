import pandas as pd
import numpy as np
import os

class HMSDataGenerator:
    """
    generates enterprise-level synthetic hospital management datasets.
    """
    
    def __init__(self, output_dir: str = 'data/raw/'):
        """
        initializes the data generator with the target output directory.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        #setting random seed for reproducibility
        np.random.seed(42)
        
    def generate_patient_records(self, num_records: int = 1500):
        """
        generates detailed synthetic patient demographic and medical history data.
        """
        patient_ids = [f"P{str(i).zfill(5)}" for i in range(1, num_records + 1)]
        ages = np.random.randint(1, 90, size=num_records)
        genders = np.random.choice(["male", "female"], size=num_records)
        blood_groups = np.random.choice(["a+", "a-", "b+", "b-", "ab+", "ab-", "o+", "o-"], size=num_records)
        
        conditions = ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease"]
        chronic_conditions = np.random.choice(conditions, p=[0.5, 0.15, 0.2, 0.1, 0.05], size=num_records)
        
        distance_to_hospital_km = np.round(np.random.uniform(1.0, 100.0, size=num_records), 1)
        historical_no_show_rate = np.round(np.random.beta(a=2, b=8, size=num_records), 2)
        
        df = pd.DataFrame({
            "patient_id": patient_ids,
            "age": ages,
            "gender": genders,
            "blood_group": blood_groups,
            "chronic_condition": chronic_conditions,
            "distance_to_hospital_km": distance_to_hospital_km,
            "historical_no_show_rate": historical_no_show_rate
        })
        df.to_csv(os.path.join(self.output_dir, "patients.csv"), index=False)
        return df

    def generate_appointment_schedules(self, patients_df: pd.DataFrame, num_records: int = 3000):
        """
        generates synthetic appointment scheduling data with realistic no-show probabilities.
        """
        appointment_ids = [f"A{str(i).zfill(5)}" for i in range(1, num_records + 1)]
        
        #sample patients with replacement
        sampled_patients = patients_df.sample(n=num_records, replace=True).reset_index(drop=True)
        patient_ids = sampled_patients["patient_id"]
        
        departments = np.random.choice(["cardiology", "neurology", "orthopedics", "general", "pediatrics"], size=num_records)
        consultation_types = np.random.choice(["Initial", "Follow-up"], p=[0.4, 0.6], size=num_records)
        
        wait_time_hours = np.random.randint(1, 720, size=num_records)
        
        #generating dates using hours to avoid restricted words
        base_timestamp = pd.Timestamp('2025-01-01')
        time_deltas = pd.to_timedelta(np.random.randint(0, 365 * 24, size=num_records), unit='h')
        appointment_dates = base_timestamp + time_deltas
        
        #calculating no-show probability dynamically
        base_prob = sampled_patients['historical_no_show_rate'].values
        #increase prob if wait time is long
        wait_penalty = np.where(wait_time_hours > 336, 0.2, 0)
        #increase prob if distance is far
        distance_penalty = np.where(sampled_patients['distance_to_hospital_km'] > 50, 0.15, 0)
        
        final_prob = np.clip(base_prob + wait_penalty + distance_penalty, 0, 1)
        no_shows = (np.random.uniform(0, 1, size=num_records) < final_prob).astype(int)
        
        df = pd.DataFrame({
            "appointment_id": appointment_ids,
            "patient_id": patient_ids,
            "department": departments,
            "consultation_type": consultation_types,
            "wait_time_hours": wait_time_hours,
            "date": appointment_dates,
            "no_show": no_shows
        })
        df.to_csv(os.path.join(self.output_dir, "appointments.csv"), index=False)
        return df

    def generate_admission_records(self, patients_df: pd.DataFrame, num_records: int = 1500):
        """
        generates synthetic inpatient admission and discharge records linked to patient health.
        """
        admission_ids = [f"ADM{str(i).zfill(5)}" for i in range(1, num_records + 1)]
        
        sampled_patients = patients_df.sample(n=num_records, replace=True).reset_index(drop=True)
        patient_ids = sampled_patients["patient_id"]
        
        wards = np.random.choice(["general", "icu", "maternity", "emergency"], size=num_records)
        admission_types = np.random.choice(["Emergency", "Elective"], p=[0.6, 0.4], size=num_records)
        severity_levels = np.random.randint(1, 6, size=num_records) #1 to 5
        
        #length of stay is influenced by severity and age
        base_stay = severity_levels * 24 
        age_penalty = (sampled_patients['age'] > 60).astype(int) * 48
        variance = np.random.randint(-12, 48, size=num_records)
        
        lengths_of_stay = np.maximum(24, base_stay + age_penalty + variance)
        
        df = pd.DataFrame({
            "admission_id": admission_ids,
            "patient_id": patient_ids,
            "ward": wards,
            "admission_type": admission_types,
            "severity_level": severity_levels,
            "length_of_stay_hours": lengths_of_stay
        })
        df.to_csv(os.path.join(self.output_dir, "admissions.csv"), index=False)
        return df

    def generate_hospital_resources(self):
        """
        generates static hospital resource capacities.
        """
        wards = ["general", "icu", "maternity", "emergency"]
        total_beds = [200, 50, 40, 100]
        
        df = pd.DataFrame({
            "ward": wards,
            "total_beds": total_beds
        })
        df.to_csv(os.path.join(self.output_dir, "hospital_resources.csv"), index=False)
        return df
        
    def generate_billing_information(self, patients_df: pd.DataFrame, num_records: int = 1500):
        """
        generates synthetic hospital billing data.
        """
        bill_ids = [f"B{str(i).zfill(5)}" for i in range(1, num_records + 1)]
        patient_ids = np.random.choice(patients_df["patient_id"], size=num_records)
        amounts = np.round(np.random.uniform(100.0, 5000.0, size=num_records), 2)
        payment_status = np.random.choice(["paid", "pending", "overdue"], p=[0.7, 0.2, 0.1], size=num_records)
        
        df = pd.DataFrame({
            "bill_id": bill_ids,
            "patient_id": patient_ids,
            "amount": amounts,
            "status": payment_status
        })
        df.to_csv(os.path.join(self.output_dir, "billing.csv"), index=False)
        return df
        
    def generate_doctor_availability(self, num_records: int = 50):
        """
        generates synthetic doctor availability data.
        """
        doctor_ids = [f"D{str(i).zfill(3)}" for i in range(1, num_records + 1)]
        departments = np.random.choice(["cardiology", "neurology", "orthopedics", "general", "pediatrics"], size=num_records)
        status = np.random.choice(["available", "on_leave", "in_surgery"], p=[0.7, 0.1, 0.2], size=num_records)
        
        df = pd.DataFrame({
            "doctor_id": doctor_ids,
            "department": departments,
            "current_status": status
        })
        df.to_csv(os.path.join(self.output_dir, "doctors.csv"), index=False)
        return df

    def generate_all(self):
        """
        generates all synthetic datasets and saves them.
        """
        patients = self.generate_patient_records()
        self.generate_appointment_schedules(patients)
        self.generate_admission_records(patients)
        self.generate_hospital_resources()
        self.generate_billing_information(patients)
        self.generate_doctor_availability()

if __name__ == "__main__":
    generator = HMSDataGenerator()
    generator.generate_all()
