from flask import Flask, render_template, request
from datetime import datetime
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("model/rf_model.pkl")

@app.route("/predict-form", methods=["GET", "POST"])
def predict_form():
    if request.method == "POST":
        data = request.form
        try:
            dt_raw = data["datetime"]
            dt = datetime.fromisoformat(dt_raw)
            readable_dt = dt.strftime("%A, %B %d, %Y – %I:%M %p")

            day_of_year = dt.timetuple().tm_yday
            time_minutes = dt.hour * 60 + dt.minute

            lat = float(data["latitude"])
            lon = float(data["longitude"])

            X = [[day_of_year, time_minutes, lon, lat]]
            pred = model.predict(X)[0]

            return render_template("predict_result.html",
                                   prediction=pred,
                                   readable_time=readable_dt)
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
    return render_template("predict_form.html")

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True)
