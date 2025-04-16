from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from model.iot_model import IOTOccupancyModel  
app = FastAPI()
occupancy_model = IOTOccupancyModel()
templates = Jinja2Templates(directory="templates")
df = pd.read_csv("data/occupancy_predictions_actual.csv")

class PredictRequest(BaseModel):
    day_of_week: int
    time_of_day: int
    latitude: float
    longitude: float

class PredictResponse(BaseModel):
    predicted_occupancy: str

@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict occupancy category for a given day, time, and location.
    """
    features = [
        [req.day_of_week, req.time_of_day, req.latitude, req.longitude]
    ]
    pred = occupancy_model.predict(features)

    if isinstance(pred, str):
        raise HTTPException(status_code=400, detail=pred)

    return PredictResponse(predicted_occupancy=pred[0])


@app.get("/historical")
async def historical(station: str, day_of_week: int):
    """
    Returns historical occupancy data for a specified station and day.
    """
    try:
        day_of_week = int(day_of_week)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid day_of_week: must be an integer"
        )    
    if 'station' not in df.columns:
        raise HTTPException(
            status_code=500, 
            detail=f"'station' column not found. Available columns: {df.columns.tolist()}"
        )
    
    if 'day_of_week' not in df.columns:
        raise HTTPException(
            status_code=500, 
            detail=f"'day_of_week' column not found. Available columns: {df.columns.tolist()}"
        )    
    subset = df[
        (df["station"] == station) & (df["day_of_week"] == day_of_week)
    ].copy()

    if subset.empty:
        raise HTTPException(
            status_code=404,
            detail="No historical data found for the given station and day."
        )    
    subset.loc[:, "hour"] = subset["time_of_day"] // 60    
    occupancy_map = {"low": 1, "medium": 2, "high": 3}    
    hourly_data = []
    
    for hour in range(24):
        hour_subset = subset[subset["hour"] == hour]
        
        if not hour_subset.empty:
            most_common = hour_subset["predicted_occupancy"].value_counts().idxmax()
            occupancy_level = occupancy_map[most_common]
        else:
            most_common = "low"
            occupancy_level = 1        
        if hour == 0:
            hour_label = "12a"
        elif hour < 12:
            hour_label = f"{hour}a"
        elif hour == 12:
            hour_label = "12p"
        else:
            hour_label = f"{hour-12}p"
        
        hourly_data.append({
            "hour": hour,
            "hour_label": hour_label,
            "occupancy": most_common,
            "occupancy_level": occupancy_level
        })    
    hours = [item["hour"] for item in hourly_data]
    hour_labels = [item["hour_label"] for item in hourly_data]
    occupancy_levels = [item["occupancy_level"] for item in hourly_data]
    occupancy_names = [item["occupancy"] for item in hourly_data]

    return {
        "hours": hours,           
        "labels": hour_labels,    
        "values": occupancy_levels, 
        "categories": occupancy_names 
    }
