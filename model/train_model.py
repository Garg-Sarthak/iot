import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv("data/synthetic_occupancy_data.csv")

X = df[["day_of_year", "time_minutes", "latitude", "longitude"]]
y = df["occupancy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

joblib.dump(rf, "model/rf_model.pkl")
print("Model trained and saved!")
