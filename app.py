from fastapi import FastAPI, HTTPException, Request, Form # Added Form
from pydantic import BaseModel
# import joblib
import pandas as pd
from datetime import datetime, timezone # Added timezone
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model.iot_model import IOTOccupancyModel
import logging # Added logging

# --- Add Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

app = FastAPI()
occupancy_model = IOTOccupancyModel()
templates = Jinja2Templates(directory="templates")

# --- Load existing data (handle potential file not found) ---
try:
    df = pd.read_csv("data/occupancy_predictions_actual.csv")
    logger.info("Loaded historical data from data/occupancy_predictions_actual.csv")
except FileNotFoundError:
    logger.warning("data/occupancy_predictions_actual.csv not found. Historical endpoint might not work.")
    df = pd.DataFrame() # Create an empty DataFrame if file doesn't exist
# --- End Load existing data ---


# --- Pydantic Models ---
class PredictRequest(BaseModel):
    day_of_week: int
    time_of_day: int
    latitude: float
    longitude: float

class PredictResponse(BaseModel):
    predicted_occupancy: str

# --- NEW: Pydantic Model for NodeMCU Data ---
class NodeMCUData(BaseModel):
    deviceId: str | None = "UnknownNodeMCU" # Added Device ID
    latitude: float
    longitude: float
    onboardCount1: int
    onboardCount2: int
    offboardCount: int
    totalOccupancy: int
# --- End NEW Model ---


# --- Existing Endpoints ---
@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    # return templates.TemplateResponse("index.html", {"request": request})
    latest_log = None
    try:
        with open("nodemcu_sensor_log.txt", "r") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                # Split timestamp and data
                timestamp_str, data_str = last_line.split(" - ", 1)
                
                # Parse the data portion safely
                try:
                    log_data = eval(data_str)  # For prototype only
                    log_data["received_time"] = datetime.fromisoformat(timestamp_str)
                    latest_log = log_data
                except Exception as e:
                    logger.error(f"Error parsing log entry: {e}")
    except Exception as e:
        logger.error(f"Error loading logs: {e}")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "latest_log": latest_log  # Now passing a dict instead of raw string
    })

@app.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # ... (your existing prediction logic remains unchanged)
    features = [
        [req.day_of_week, req.time_of_day, req.latitude, req.longitude]
    ]
    pred = occupancy_model.predict(features)

    if isinstance(pred, str):
        raise HTTPException(status_code=400, detail=pred)

    return PredictResponse(predicted_occupancy=pred[0])


@app.get("/historical")
async def historical(station: str, day_of_week: int):
    # ... (your existing historical data logic remains unchanged)
    # Added checks for empty df
    if df.empty:
         raise HTTPException(
            status_code=404,
            detail="Historical data file not loaded or empty."
        )
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
# New endpoint to get latest log
@app.get("/latest_log")
def get_latest_log():
    try:
        with open("nodemcu_sensor_log.txt", "r") as f:
            lines = f.readlines()
            if not lines:
                return {"message": "No logs found"}
            last_line = lines[-1].strip()
            return {"latest_log": last_line}
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return {"error": "Could not retrieve logs"}

# --- NEW Endpoint to receive NodeMCU data ---
@app.post("/log_data")
async def receive_nodemcu_log(data: NodeMCUData, request: Request):
    """
    Receives sensor data from NodeMCU (sent as JSON) and logs it.
    """
    client_host = request.client.host
    received_time = datetime.now(timezone.utc)

    logger.info(f"Received data from NodeMCU {data.deviceId} ({client_host}) at {received_time}:")
    logger.info(f"  Latitude: {data.latitude}")
    logger.info(f"  Longitude: {data.longitude}")
    logger.info(f"  Onboard Count 1: {data.onboardCount1}")
    logger.info(f"  Onboard Count 2: {data.onboardCount2}")
    logger.info(f"  Offboard Count: {data.offboardCount}")
    logger.info(f"  Total Occupancy: {data.totalOccupancy}")

    # --- Add your persistence logic here ---
    # Option 1: Log to a simple text file
    try:
        with open("nodemcu_sensor_log.txt", "a") as f:
            # Log as JSON string or customize format
            f.write(f"{received_time.isoformat()} - {data.dict()}\n")
        logger.info("Successfully wrote data to nodemcu_sensor_log.txt")
    except Exception as e:
        logger.error(f"Failed to write to log file: {e}")
        # Optional: raise HTTPException(status_code=500, detail="Server failed to log data")
    

    # Option 2: Store in a database (requires setting up DB connection, e.g., using SQLAlchemy or databases library)
    # await database.execute("INSERT INTO sensor_logs(...) VALUES (...)")

    # Option 3: Append to a CSV file (less robust than DB but simple)
    # try:
    #     log_entry = {
    #         'timestamp': received_time.isoformat(),
    #         'deviceId': data.deviceId,
    #         'latitude': data.latitude,
    #         'longitude': data.longitude,
    #         'onboardCount1': data.onboardCount1,
    #         'onboardCount2': data.onboardCount2,
    #         'offboardCount': data.offboardCount,
    #         'totalOccupancy': data.totalOccupancy
    #     }
    #     log_df = pd.DataFrame([log_entry])
    #     log_df.to_csv("nodemcu_log.csv", mode='a', header=not pd.io.common.file_exists("nodemcu_log.csv"), index=False)
    #     logger.info("Successfully appended data to nodemcu_log.csv")
    # except Exception as e:
    #     logger.error(f"Failed to write to CSV file: {e}")


    return {"status": "success", "message": "Data logged"}
# --- End NEW Endpoint ---


# --- Optional: Endpoint to handle the old form-urlencoded data ---
# If you still need to support the old POST request format to /send for some reason
# @app.post("/send")
# async def receive_nodemcu_form_log(
#     request: Request,
#     lat: float = Form(...),
#     long: float = Form(...), # Note the key name from NodeMCU code
#     onboard: int = Form(...),
#     offboard: int = Form(...),
#     total: int = Form(...)
# ):
#     """
#     Receives sensor data from NodeMCU (sent as x-www-form-urlencoded)
#     """
#     client_host = request.client.host
#     received_time = datetime.now(timezone.utc)
#     logger.info(f"Received FORM data from NodeMCU ({client_host}) at {received_time}:")
#     logger.info(f"  Latitude: {lat}")
#     logger.info(f"  Longitude: {long}")
#     logger.info(f"  Onboard Total: {onboard}") # This was onboardTotal
#     logger.info(f"  Offboard Count: {offboard}")
#     logger.info(f"  Total Occupancy: {total}")

#     # Log this data similarly (e.g., to a different file or add a source field)
#     try:
#         with open("nodemcu_form_log.txt", "a") as f:
#             f.write(f"{received_time.isoformat()} - lat={lat}, long={long}, onboard={onboard}, offboard={offboard}, total={total}\n")
#         logger.info("Successfully wrote form data to nodemcu_form_log.txt")
#     except Exception as e:
#         logger.error(f"Failed to write form data to log file: {e}")

#     return {"status": "success", "message": "Form data received"}

# --- Run instruction (as comment) ---
# To run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
