import os
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import numpy as np
from supabase import create_client, Client

app = FastAPI(title="Breast Cancer Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Optional[Client] = None

# Global variables to store model and scaler
model = None
scaler = None

class PredictionInput(BaseModel):
    features: List[float]
    modelBucket: str
    modelFile: str
    scalerFile: str

class PredictionOutput(BaseModel):
    prediction: str
    probability: float

@app.on_event("startup")
async def startup_db_client():
    global supabase
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.")
    supabase = create_client(supabase_url, supabase_key)
    print("Supabase client initialized successfully")

def download_and_load_model(bucket_name: str, model_file: str, scaler_file: str):
    global model, scaler
    
    # Download model file
    try:
        print(f"Downloading model file: {model_file} from bucket: {bucket_name}")
        model_response = supabase.storage.from_(bucket_name).download(model_file)
        model = pickle.loads(model_response)
        print("Model loaded successfully")
        
        # Download scaler file
        print(f"Downloading scaler file: {scaler_file} from bucket: {bucket_name}")
        scaler_response = supabase.storage.from_(bucket_name).download(scaler_file)
        scaler = pickle.loads(scaler_response)
        print("Scaler loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        return False

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    global model, scaler
    
    # Check if model and scaler need to be loaded
    if model is None or scaler is None:
        success = download_and_load_model(
            input_data.modelBucket,
            input_data.modelFile,
            input_data.scalerFile
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load model or scaler")
    
    try:
        # Convert features to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Get prediction
        prediction_code = model.predict(scaled_features)[0]
        
        # Get probability
        proba = model.predict_proba(scaled_features)[0]
        probability = proba[1] if prediction_code == 1 else proba[0]
        
        # Map prediction code to text
        prediction_text = "Malignant" if prediction_code == 1 else "Benign"
        
        return {
            "prediction": prediction_text,
            "probability": float(probability)
        }
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "scaler_loaded": scaler is not None}

if __name__ == "__main__":
    # Get port from environment variable for Heroku compatibility
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)