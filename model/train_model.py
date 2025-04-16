import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

df = pd.read_csv(r'C:\Users\Vedant\Desktop\IOTPROJECT\data\dataset.csv')
print("Columns:", df.columns.tolist())

X = df[["DayOfWeek", "TimeOfDayMinutes", "Latitude", "Longitude"]]
y = df["Occupancy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
os.makedirs("model", exist_ok=True)
joblib.dump(tree_model, "model/decision_tree_model.pkl")

print("Decision Tree model trained and saved successfully!")
