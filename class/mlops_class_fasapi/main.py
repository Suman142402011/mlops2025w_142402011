import os
import joblib
import wandb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist, validator
import numpy as np

app = FastAPI()
MODEL_ARTIFACT = os.environ.get(
    'WANDER_MODEL_ARTIFACT',
    '142402011-indian-institute-of-technology/classroom-deploy/iris-rf:v0'
)

# -----------------
# Pydantic models
# -----------------
# Use conlist to require a list of floats. Here min_items=1 keeps it flexible;
# change to min_items=4, max_items=4 if you want to enforce exactly 4 features (Iris).
class PredictRequest(BaseModel):
    features: conlist(float, min_length=4)

    @validator("features")
    def no_nan(cls, v):
        if any(np.isnan(x) for x in v):
            raise ValueError("features must not contain NaN")
        return v


class PredictResponse(BaseModel):
    prediction: int


# -----------------
# Model loader
# -----------------
def load_model_from_wandb(artifact_ref):
    try:
        wandb.login()
    except Exception:
        pass
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    path = artifact.download()
    model_file = os.path.join(path, 'model.pkl')
    return joblib.load(model_file)


# -----------------
# Startup
# -----------------
@app.on_event('startup')
def startup():
    global model
    model = load_model_from_wandb(MODEL_ARTIFACT)


# -----------------
# Endpoints
# -----------------
@app.get('/')
def root():
    return {'status': 'ok', 'model_artifact': MODEL_ARTIFACT}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Accepts JSON body validated by Pydantic, e.g.:
      {"features": [5.1, 3.5, 1.4, 0.2]}
    """
    try:
        arr = np.array(req.features, dtype=float).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid features: {e}")

    try:
        pred = model.predict(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Ensure the prediction is JSON-serializable (int)
    pred_val = int(pred[0]) if hasattr(pred[0], "item") or isinstance(pred[0], (int, float)) else pred[0]
    return PredictResponse(prediction=pred_val)
