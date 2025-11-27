from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import pandas as pd
import os
from datetime import datetime
import logging
from supabase import create_client, Client

from simulation_gnn_model import HazeSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HazeRadar Simulation API",
    description="GNN-based haze propagation simulation for Indonesia",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "simulation_gcn_best.pt")

simulator: Optional[HazeSimulator] = None
supabase_client: Optional[Client] = None


class FireZoneCoordinate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class SimulationRequest(BaseModel):
    fire_zone_coords: List[FireZoneCoordinate]
    simulation_hours: int = Field(default=72, ge=1, le=168)
    radius_km: float = Field(default=50.0, ge=10, le=200)

    @validator('fire_zone_coords')
    def validate_coords(cls, v):
        if len(v) == 0:
            raise ValueError("At least one coordinate must be provided")
        if len(v) > 50:
            raise ValueError("Maximum 50 coordinates allowed")
        return v


class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    message: str
    total_predictions: int
    forecast_hours: int
    predictions: List[Dict]


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str


def get_supabase_client() -> Client:
    global supabase_client
    if supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise HTTPException(status_code=500, detail="Supabase credentials not configured")
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase_client


def get_simulator() -> HazeSimulator:
    global simulator
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return simulator


@app.on_event("startup")
async def startup_event():
    global simulator
    try:
        logger.info("Starting HazeRadar Simulation API...")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        simulator = HazeSimulator(MODEL_PATH)
        simulator.load_model(in_feats=8)
        
        supabase = get_supabase_client()
        
        city_graph_response = supabase.table("city_graph_structure").select("*").execute()
        city_graph = city_graph_response.data
        logger.info(f"Loaded {len(city_graph)} city connections")
        
        training_data_response = supabase.table("gnn_training_data").select("*").execute()
        training_data = pd.DataFrame(training_data_response.data)
        
        training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])
        num_cols = [
            "latitude", "longitude", "temperature", "humidity", "wind_speed",
            "avg_fire_confidence", "upwind_fire_count", "current_aqi",
            "population_density"
        ]
        training_data[num_cols] = training_data[num_cols].apply(pd.to_numeric, errors='coerce')
        training_data.dropna(inplace=True)
        
        logger.info(f"Loaded {len(training_data)} training data rows")
        
        simulator.initialize_graph(city_graph, training_data)
        
        logger.info("Simulator initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        model_loaded=simulator is not None,
        database_connected=supabase_client is not None,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/cities")
async def get_cities(sim: HazeSimulator = Depends(get_simulator)):
    try:
        supabase = get_supabase_client()
        response = supabase.table("gnn_training_data").select("city, latitude, longitude").execute()
        
        cities_df = pd.DataFrame(response.data)
        cities_unique = cities_df.groupby('city').first().reset_index()
        
        return {
            "total_cities": len(cities_unique),
            "cities": cities_unique.to_dict('records')
        }
    except Exception as e:
        logger.error(f"Failed to fetch cities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate_haze(
    request: SimulationRequest,
    sim: HazeSimulator = Depends(get_simulator)
):
    try:
        logger.info(f"Starting simulation with {len(request.fire_zone_coords)} fire zones")
        
        supabase = get_supabase_client()
        response = supabase.table("gnn_training_data").select("*").execute()
        training_data = pd.DataFrame(response.data)
        
        training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])
        num_cols = [
            "latitude", "longitude", "temperature", "humidity", "wind_speed",
            "wind_direction", "avg_fire_confidence", "upwind_fire_count", 
            "current_aqi", "population_density"
        ]
        training_data[num_cols] = training_data[num_cols].apply(pd.to_numeric, errors='coerce')
        training_data.dropna(inplace=True)
        
        fire_coords = [coord.dict() for coord in request.fire_zone_coords]
        
        result_df = sim.simulate_fire_zone(
            fire_zone_coords=fire_coords,
            initial_features=training_data,
            hours=request.simulation_hours
        )
        
        predictions = result_df.to_dict('records')
        for pred in predictions:
            if 'timestamp' in pred and isinstance(pred['timestamp'], pd.Timestamp):
                pred['timestamp'] = pred['timestamp'].isoformat()
        
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return SimulationResponse(
            simulation_id=simulation_id,
            status="success",
            message=f"Simulation completed for {request.simulation_hours} hours",
            total_predictions=len(predictions),
            forecast_hours=request.simulation_hours,
            predictions=predictions
        )
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.get("/api/simulation/{simulation_id}")
async def get_simulation_results(simulation_id: str):
    return {
        "simulation_id": simulation_id,
        "status": "completed",
        "message": "Use POST /api/simulate to run new simulations"
    }


@app.get("/api/model/info")
async def get_model_info(sim: HazeSimulator = Depends(get_simulator)):
    return {
        "model_type": "Graph Convolutional Network (GCN)",
        "input_features": sim.feature_cols,
        "num_cities": len(sim.node_cities),
        "num_edges": sim.edge_index.shape[1] if sim.edge_index is not None else 0,
        "device": str(sim.device)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
