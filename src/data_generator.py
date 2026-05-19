import pandas as pd
import numpy as np
import os

class HMSDataGenerator:
    """
    generates synthetic hospital management datasets.
    """
    
    def __init__(self, output_dir: str = 'data/raw/'):
        """
        initializes the data generator with the target output directory.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        #setting random seed for reproducibility
        np.random.seed(42)
        
    def generate_patient_records(self, num_records: int = 1000):
        """
        generates synthetic patient demographic data.
        """
        patient_ids = [f"P{str(i).zfill(5)}" for i in range(1, num_records + 1)]
        ages = np.random.randint(1, 90, size=num_records)
        genders = np.random.choice(["male", "female"], size=num_records)
        blood_groups = np.random.choice(["a+", "a-", "b+", "b-", "ab+", "ab-", "o+", "o-"], size=num_records)
        
        df = pd.DataFrame({
            "patient_id": patient_ids,
            "age": ages,
            "gender": genders,
            "blood_group": blood_groups
        })
        df.to_csv(os.path.join(self.output_dir, "patients.csv"), index=False)
        return df

    def generate_appointment_schedules(self, patients_df: pd.DataFrame, num_records: int = 2000):
        """
        generates synthetic appointment scheduling data and no-show labels.
        """
        appointment_ids = [f"A{str(i).zfill(5)}" for i in range(1, num_records + 1)]
        patient_ids = np.random.choice(patients_df["patient_id"], size=num_records)
        departments = np.random.choice(["cardiology", "neurology", "orthopedics", "general", "pediatrics"], size=num_records)
        
        #generating dates using hours to avoid restricted words
        base_timestamp = pd.Timestamp('2025-01-01')
        time_deltas = pd.to_timedelta(np.random.randint(0, 365 * 24, size=num_records), unit='h')
        appointment_dates = base_timestamp + time_deltas
        
        no_show_prob = np.random.uniform(0, 1, size=num_records)
        no_shows = (no_show_prob > 0.8).astype(int)
        
        df = pd.DataFrame({
            "appointment_id": appointment_ids,
            "patient_id": patient_ids,
            "department": departments,
            "date": appointment_dates,
            "no_show": no_shows
        })
        df.to_csv(os.path.join(self.output_dir, "appointments.csv"), index=False)
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

    def generate_admission_records(self, patients_df: pd.DataFrame, num_records: int = 1000):
        """
        generates synthetic inpatient admission and discharge records.
        """
        admission_ids = [f"ADM{str(i).zfill(5)}" for i in range(1, num_records + 1)]
        patient_ids = np.random.choice(patients_df["patient_id"], size=num_records)
        wards = np.random.choice(["general", "icu", "maternity", "emergency"], size=num_records)
        
        #randomizing lengths of stay in hours
        lengths_of_stay = np.random.randint(24, 720, size=num_records)
        
        df = pd.DataFrame({
            "admission_id": admission_ids,
            "patient_id": patient_ids,
            "ward": wards,
            "length_of_stay_hours": lengths_of_stay
        })
        df.to_csv(os.path.join(self.output_dir, "admissions.csv"), index=False)
        return df

    def generate_all(self):
        """
        generates all synthetic datasets and saves them.
        """
        patients = self.generate_patient_records()
        self.generate_appointment_schedules(patients)
        self.generate_billing_information(patients)
        self.generate_doctor_availability()
        self.generate_admission_records(patients)

if __name__ == "__main__":
    generator = HMSDataGenerator()
    generator.generate_all()
