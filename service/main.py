from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
import joblib
import numpy as np
from typing import List

app = FastAPI()
model: IsolationForest = joblib.load("model.joblib")


class PredictItem(BaseModel):
    feature_vector: List[float]
    score: Optional[bool]


@app.get("/model_information")
async def model_information():
    global model;

    return model.get_params()


@app.post("/predict")
async def predict(item: PredictItem):
    global model
    response = {}
    print("Predicting")
    features = np.array(item.feature_vector).reshape(-1, 2)
    print(features)
    res = model.predict(features)
    response["is_inloer"] = res[0].item()
    if item.score:
        res = model.score_samples(features)
        response["anomaly_score"] = res[0]

    return response


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    None

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
