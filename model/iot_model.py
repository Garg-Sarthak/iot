import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class IOTOccupancyModel:
    def __init__(self):
        try:
            self.model = joblib.load("model/decision_tree_model.pkl")
        except FileNotFoundError:
            print("Model not trained yet. Please train the model first.")
            self.model = None
    def predict(self, features):
        if self.model is None:
            return "No model loaded."
        df = pd.DataFrame(features, columns=["DayOfWeek", "TimeOfDayMinutes", "Latitude", "Longitude"])
        return self.model.predict(df)
