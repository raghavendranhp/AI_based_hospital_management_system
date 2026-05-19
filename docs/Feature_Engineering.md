# Feature Engineering Proposal

## Overview
To effectively train the predictive models, the raw data must be transformed into a format suitable for machine learning algorithms (Random Forest & XGBoost).

## Proposed Transformations

### 1. No-Show Prediction
- **Target Variable**: `no_show` (Binary: 0 or 1)
- **Input Feature**: `department`
- **Transformation**: Categorical encoding (Dummy encoding / Categorical codes). 
- **Future Enhancements**: Extracting the hour, month, or day-of-week from the `date` timestamp to capture temporal trends in missed appointments. Including patient demographics (age, gender) by merging `appointments.csv` with `patients.csv`.

### 2. Bed Occupancy Prediction
- **Target Variable**: `length_of_stay_hours` (Continuous integer)
- **Input Feature**: `ward`
- **Transformation**: Categorical encoding to convert ward names (e.g., ICU, general, maternity) into numeric codes.
- **Future Enhancements**: Using admission diagnosis codes (if available) and patient age to better predict the length of stay, as older patients typically require longer hospitalizations.

## Tools
- `pandas` for merging and transformations.
- `sklearn.preprocessing` for robust encoding pipelines.
