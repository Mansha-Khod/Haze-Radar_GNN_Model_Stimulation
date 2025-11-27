import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationGCN(torch.nn.Module):
    def __init__(self, in_feats, hidden=64, out_feats=1, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, out_feats)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x


class HazeSimulator:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.node_cities = []
        self.city_to_idx = {}
        self.edge_index = None
        self.feature_cols = [
            "temperature", "humidity", "wind_speed", "wind_direction",
            "avg_fire_confidence", "upwind_fire_count", "current_aqi", "population_density"
        ]
        self.model_path = model_path
        logger.info(f"Initializing HazeSimulator on device: {self.device}")

    def load_model(self, in_feats: int = 8):
        try:
            self.model = SimulationGCN(in_feats=in_feats, hidden=64, out_feats=1, dropout=0.2)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def initialize_graph(self, city_graph: List[Dict], node_data: pd.DataFrame):
        try:
            self.node_cities = list(node_data['city'].unique())
            self.city_to_idx = {c: i for i, c in enumerate(self.node_cities)}
            
            edges = []
            for connection in city_graph:
                src_city = connection.get('city')
                src = self.city_to_idx.get(src_city)
                if src is None:
                    continue
                
                connected = connection.get('connected_cities', [])
                if isinstance(connected, str):
                    connected = [c.strip() for c in connected.split(',')]
                
                for c in connected:
                    c = str(c).strip()
                    if c == "" or c.lower() == 'nan':
                        continue
                    dst = self.city_to_idx.get(c)
                    if dst is not None:
                        edges.append([src, dst])
            
            if len(edges) == 0:
                raise ValueError("No valid edges created from city graph")
            
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            logger.info(f"Graph initialized with {len(self.node_cities)} nodes and {len(edges)} edges")
        except Exception as e:
            logger.error(f"Failed to initialize graph: {str(e)}")
            raise

    def prepare_initial_features(self, node_data: pd.DataFrame) -> torch.Tensor:
        latest_per_city = node_data.sort_values('timestamp').groupby('city', as_index=False).last()
        latest_per_city = latest_per_city.set_index('city').reindex(self.node_cities).reset_index()
        
        X = latest_per_city[self.feature_cols].values.astype(np.float32)
        return torch.tensor(X).to(self.device)

    def simulate_fire_zone(self, fire_zone_coords: List[Dict[str, float]], 
                          initial_features: pd.DataFrame,
                          hours: int = 72) -> pd.DataFrame:
        try:
            current_x = self.prepare_initial_features(initial_features)
            init_df = initial_features.sort_values('timestamp').groupby('city', as_index=False).last()
            init_df = init_df.set_index('city').reindex(self.node_cities).reset_index()
            
            affected_nodes = self._find_affected_nodes(fire_zone_coords, init_df)
            
            update_idx = self.feature_cols.index("avg_fire_confidence")
            fire_idx = self.feature_cols.index("upwind_fire_count")
            
            current_x_np = current_x.cpu().numpy()
            for node_idx in affected_nodes:
                current_x_np[node_idx, update_idx] = 85.0
                current_x_np[node_idx, fire_idx] = 10.0
            current_x = torch.tensor(current_x_np.astype(np.float32)).to(self.device)
            
            future_predictions = []
            aqi_update_idx = self.feature_cols.index("current_aqi")
            
            for h in range(1, hours + 1):
                with torch.no_grad():
                    data = Data(x=current_x, edge_index=self.edge_index)
                    pred = self.model(data).cpu().numpy().flatten()
                
                pred = np.clip(pred, 0, 500)
                
                hour_df = pd.DataFrame({
                    "city": self.node_cities,
                    "latitude": init_df["latitude"].values,
                    "longitude": init_df["longitude"].values,
                    "wind_speed": init_df["wind_speed"].values,
                    "wind_direction": init_df["wind_direction"].values,
                    "predicted_pm25": pred
                })
                
                max_pm25 = max(hour_df["predicted_pm25"].max(), 1.0)
                hour_df["haze_intensity"] = (hour_df["predicted_pm25"] / max_pm25) * 100
                hour_df["aqi_category"] = hour_df["predicted_pm25"].apply(self._pm25_to_aqi_category)
                hour_df["timestamp"] = datetime.now() + timedelta(hours=h)
                hour_df["forecast_hour"] = h
                
                future_predictions.append(hour_df)
                
                next_x = current_x.clone().cpu().numpy()
                next_x[:, aqi_update_idx] = pred
                
                decay_factor = 0.95
                next_x[:, update_idx] *= decay_factor
                
                current_x = torch.tensor(next_x.astype(np.float32)).to(self.device)
            
            result_df = pd.concat(future_predictions, ignore_index=True)
            logger.info(f"Simulation completed: {len(result_df)} predictions generated")
            return result_df
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise

    def _find_affected_nodes(self, fire_zone_coords: List[Dict[str, float]], 
                            node_df: pd.DataFrame, radius_km: float = 50.0) -> List[int]:
        affected = []
        for coord in fire_zone_coords:
            lat, lon = coord['latitude'], coord['longitude']
            for idx, row in node_df.iterrows():
                dist = self._haversine_distance(lat, lon, row['latitude'], row['longitude'])
                if dist <= radius_km:
                    affected.append(idx)
        return list(set(affected))

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    @staticmethod
    def _pm25_to_aqi_category(pm25: float) -> str:
        if pm25 <= 12.0:
            return "Good"
        elif pm25 <= 35.4:
            return "Moderate"
        elif pm25 <= 55.4:
            return "Unhealthy for Sensitive Groups"
        elif pm25 <= 150.4:
            return "Unhealthy"
        elif pm25 <= 250.4:
            return "Very Unhealthy"
        else:
            return "Hazardous"
