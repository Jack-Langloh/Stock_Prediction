import json
import os
import numpy as np
import pandas as pd
from joblib import load

# SageMaker will place model files under /opt/ml/model/
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "finalized_model.joblib")
    model = load(model_path)
    return model

def input_fn(request_body, request_content_type):
    # Accept JSON or CSV
    if request_content_type == "application/json":
        payload = json.loads(request_body)

        # payload can be {"data": [...] } or just a list
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]

        return pd.DataFrame(payload)

    if request_content_type == "text/csv":
        from io import StringIO
        return pd.read_csv(StringIO(request_body), header=None)

    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    preds = model.predict(input_data)
    return preds

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        return json.dumps({"predictions": prediction.tolist()}), response_content_type

    # default
    return json.dumps({"predictions": prediction.tolist()}), "application/json"
