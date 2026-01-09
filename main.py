import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from supabase import create_client, Client

from simulation_gnn_model import HazeSimulator

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("haze-radar")

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Haze Radar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Globals
# ---------------------------
supabase: Optional[Client] = None
model: Optional[HazeSimulator] = None


# ---------------------------
# Lazy Supabase connection
# ---------------------------
def get_supabase() -> Client:
    global supabase

    if supabase is not None:
        return supabase

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("Supabase environment variables are missing")

    supabase = create_client(url, key)
    logger.info("Supabase client initialized")
    return supabase


# ---------------------------
# Startup event
# ---------------------------
@app.on_event("startup")
def startup_event():
    global model
    logger.info("Starting application")

    model = HazeSimulator()
    logger.info("Haze GNN model loaded")


# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def root():
    return {"status": "Haze Radar backend is running"}


# ---------------------------
# Request Models
# ---------------------------
class SimulationRequest(BaseModel):
    city: str
    humidity: float
    temperature: float
    wind_speed: float
    pm25: float
    pm10: float


# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/simulate")
def simulate(req: SimulationRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        result = model.run_simulation(
            city=req.city,
            humidity=req.humidity,
            temperature=req.temperature,
            wind_speed=req.wind_speed,
            pm25=req.pm25,
            pm10=req.pm10
        )

        return {
            "city": req.city,
            "prediction": result,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.exception("Simulation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/store")
def store_simulation(req: SimulationRequest):
    try:
        sb = get_supabase()

        if model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        prediction = model.run_simulation(
            city=req.city,
            humidity=req.humidity,
            temperature=req.temperature,
            wind_speed=req.wind_speed,
            pm25=req.pm25,
            pm10=req.pm10
        )

        payload = {
            "city": req.city,
            "humidity": req.humidity,
            "temperature": req.temperature,
            "wind_speed": req.wind_speed,
            "pm25": req.pm25,
            "pm10": req.pm10,
            "prediction": prediction,
            "created_at": datetime.utcnow().isoformat()
        }

        response = sb.table("haze_predictions").insert(payload).execute()

        return {
            "status": "stored",
            "data": response.data
        }

    except Exception as e:
        logger.exception("Supabase insert failed")
        raise HTTPException(status_code=500, detail=str(e))
