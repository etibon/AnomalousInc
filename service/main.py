from typing import Optional
import time
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np
from typing import List
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge

app = FastAPI()
model: IsolationForest = joblib.load("model.joblib")
metrics_app = prometheus_client.make_asgi_app()
app.mount("/metrics", metrics_app)
model_information_counter = Counter('model_information_calls', 'Calls to /model_information endpoint')
predict_counter = Counter('predict_calls', 'Calls to /predict endpoint')
predict_response_hist = Histogram('predict_response_hist', 'Prediction response histogram')
predict_score_hist = Histogram("predict_score_hist", "Prediction score histogram")
predict_latency = Histogram('predict_latency', 'Prediction latency')
predict_label = Gauge("predict_label", "Prediction label")

class PredictItem(BaseModel):
    feature_vector: List[float]
    score: Optional[bool]


@app.get("/model_information")
async def model_information():
    global model;

    model_information_counter.inc(1)

    return model.get_params()


@app.post("/predict")
async def predict(item: PredictItem):
    global model

    start = time.perf_counter()

    predict_counter.inc(1)

    response = {}
    print("Predicting")
    features = np.array(item.feature_vector).reshape(-1, 2)
    print(features)
    res = model.predict(features)
    response["is_inlier"] = res[0].item()
    predict_response_hist.observe(res[0].item())
    if res[0].item() < 0 :
        predict_label.dec()
    else:
        predict_label.inc()
    if item.score:
        res = model.score_samples(features)
        response["anomaly_score"] = res[0]
        predict_score_hist.observe(res[0])

    end = time.perf_counter()
    elapsed = end-start
    predict_latency.observe(elapsed)

    return response


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    None

