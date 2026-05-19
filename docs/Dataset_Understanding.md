# Dataset Understanding Report

## Overview
The dataset consists of 5 synthetic, interrelated data tables representing various facets of hospital operations.

## Datasets

### 1. Patients (`patients.csv`)
- **Features**: `patient_id`, `age`, `gender`, `blood_group`
- **Volume**: 1,000 records
- **Purpose**: Demographic base for all other operations.

### 2. Appointments (`appointments.csv`)
- **Features**: `appointment_id`, `patient_id`, `department`, `date`, `no_show`
- **Volume**: 2,000 records
- **Purpose**: Tracks scheduled visits and whether the patient attended or missed the appointment (`no_show`).

### 3. Admissions (`admissions.csv`)
- **Features**: `admission_id`, `patient_id`, `ward`, `length_of_stay_hours`
- **Volume**: 1,000 records
- **Purpose**: Captures inpatient data for bed occupancy tracking and forecasting.

### 4. Billing (`billing.csv`)
- **Features**: `bill_id`, `patient_id`, `amount`, `status`
- **Volume**: 1,500 records
- **Purpose**: Financial tracking.

### 5. Doctors (`doctors.csv`)
- **Features**: `doctor_id`, `department`, `current_status`
- **Volume**: 50 records
- **Purpose**: Tracks doctor availability across departments.

## Data Quality
Since the data is synthetically generated via `HMSDataGenerator`, there are no missing values or extreme outliers. The distributions are controlled to provide realistic modeling targets.
