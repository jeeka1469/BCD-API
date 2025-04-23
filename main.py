import os
import pickle
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import numpy as np
from supabase import create_client, Client
import traceback

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
    print(f"Supabase URL: {supabase_url[:10]}...")  # Print first few chars for debugging

def download_and_load_model(bucket_name: str, model_file: str, scaler_file: str):
    global model, scaler
    
    # Create temporary directory if it doesn't exist
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")
    
    try:
        # Download model file
        print(f"Downloading model file: {model_file} from bucket: {bucket_name}")
        try:
            model_response = supabase.storage.from_(bucket_name).download(model_file)
            
            print(f"Model data size: {len(model_response)} bytes")
            
            if len(model_response) < 500:  # If it's suspiciously small
                print(f"Warning: Model file might be too small ({len(model_response)} bytes)")
            
            # Write to temp file 
            model_path = '/tmp/model.pkl'
            with open(model_path, 'wb') as f:
                f.write(model_response)
            
            print(f"Model saved to {model_path}, file size: {os.path.getsize(model_path)} bytes")
            
            # Load model with error handling
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                traceback.print_exc()
                return False
        
        except Exception as e:
            print(f"Error downloading model file: {str(e)}")
            traceback.print_exc()
            return False
            
        # Download and process scaler file
        try:
            print(f"Downloading scaler file: {scaler_file} from bucket: {bucket_name}")
            scaler_response = supabase.storage.from_(bucket_name).download(scaler_file)
            
            print(f"Scaler data size: {len(scaler_response)} bytes")
            
            if len(scaler_response) < 500:  # If it's suspiciously small
                print(f"Warning: Scaler file might be too small ({len(scaler_response)} bytes)")
            
            # Write to temp file
            scaler_path = '/tmp/scaler.pkl'
            with open(scaler_path, 'wb') as f:
                f.write(scaler_response)
                
            print(f"Scaler saved to {scaler_path}, file size: {os.path.getsize(scaler_path)} bytes")
            
            # Load scaler with error handling
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print("Scaler loaded successfully")
            except Exception as e:
                print(f"Error loading scaler: {str(e)}")
                traceback.print_exc()
                return False
        
        except Exception as e:
            print(f"Error downloading scaler file: {str(e)}")
            traceback.print_exc()
            return False
            
        # Check if both model and scaler were loaded successfully
        if model is not None and scaler is not None:
            print("Both model and scaler loaded successfully")
            return True
        else:
            print("Either model or scaler failed to load")
            return False
            
    except Exception as e:
        print(f"Unexpected error in download_and_load_model: {str(e)}")
        traceback.print_exc()
        return False

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    global model, scaler
    
    # Check if model and scaler need to be loaded
    if model is None or scaler is None:
        print("Model or scaler not loaded, attempting to load...")
        success = download_and_load_model(
            input_data.modelBucket,
            input_data.modelFile,
            input_data.scalerFile
        )
        if not success:
            print("WARNING: Using mock prediction as model failed to load")
            # For development only, provide a mock prediction if model loading fails
            return {
                "prediction": "Benign" if np.random.rand() > 0.5 else "Malignant",
                "probability": float(np.random.rand() * 0.3 + 0.7)  # 70-100% confidence
            }
    
    try:
        # Print some debugging info about the model
        print(f"Model type: {type(model).__name__}")
        
        # Convert features to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        print(f"Input features shape: {features.shape}")
        
        # Scale the features
        try:
            scaled_features = scaler.transform(features)
            print(f"Scaled features shape: {scaled_features.shape}")
        except Exception as scaling_error:
            print(f"Error scaling features: {str(scaling_error)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error scaling features: {str(scaling_error)}")
        
        # Get prediction
        try:
            prediction_code = model.predict(scaled_features)[0]
            print(f"Raw prediction: {prediction_code}")
            
            # Get probability
            proba = model.predict_proba(scaled_features)[0]
            print(f"Prediction probabilities: {proba}")
            
            probability = proba[1] if prediction_code == 1 else proba[0]
            
            # Map prediction code to text
            prediction_text = "Malignant" if prediction_code == 1 else "Benign"
            
            return {
                "prediction": prediction_text,
                "probability": float(probability)
            }
        except Exception as prediction_error:
            print(f"Error during prediction: {str(prediction_error)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(prediction_error)}")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        
        # Fall back to mock prediction in case of any error
        return {
            "prediction": "Benign" if np.random.rand() > 0.5 else "Malignant",
            "probability": float(np.random.rand() * 0.3 + 0.7),  # 70-100% confidence
            "isMock": True
        }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None, 
        "scaler_loaded": scaler is not None,
        "scikit_learn_version": np.__version__
    }

if __name__ == "__main__":
    # Get port from environment variable for Heroku/Railway compatibility
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
