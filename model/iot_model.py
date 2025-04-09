import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class IOTOccupancyModel:
    def __init__(self):
        try:
            self.model = joblib.load("model/rf_model.pkl")
        except FileNotFoundError:
            print("❌ Model not trained yet. Please train the model first.")

    def predict(self, features):
        df = pd.DataFrame(features, columns=["day_of_year", "time_minutes", "latitude", "longitude"])
        return self.model.predict(df)
