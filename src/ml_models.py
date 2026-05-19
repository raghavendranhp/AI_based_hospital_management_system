import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class HMSPredictor:
    """
    machine learning predictor for hospital management tasks.
    """

    def __init__(self):
        """
        initializes the models.
        """
        self.no_show_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.occupancy_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train_no_show_model(self, data: pd.DataFrame):
        """
        trains the appointment no-show prediction model.
        """
        #basic feature engineering
        data['department_code'] = data['department'].astype('category').cat.codes
        
        features = ['department_code']
        target = 'no_show'
        
        x = data[features]
        y = data[target]
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.no_show_model.fit(x_train, y_train)
        
        preds = self.no_show_model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        return acc
        
    def predict_no_show(self, input_features: pd.DataFrame):
        """
        predicts no-show probability for given appointments.
        """
        return self.no_show_model.predict(input_features)

    def train_occupancy_model(self, data: pd.DataFrame):
        """
        trains the bed occupancy prediction model.
        """
        #simple setup using lengths of stay
        data['ward_code'] = data['ward'].astype('category').cat.codes
        
        features = ['ward_code']
        target = 'length_of_stay_hours'
        
        x = data[features]
        y = data[target]
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.occupancy_model.fit(x_train, y_train)
        
        preds = self.occupancy_model.predict(x_test)
        mse = mean_squared_error(y_test, preds)
        return mse

    def predict_occupancy(self, input_features: pd.DataFrame):
        """
        predicts occupancy lengths.
        """
        return self.occupancy_model.predict(input_features)
