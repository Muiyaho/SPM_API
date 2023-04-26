from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import statsmodels.api as sm

app = FastAPI()

class TimeSeriesData(BaseModel):
    date: str
    value: float

@app.post("/predict_ses")
async def predict_ses(data: List[TimeSeriesData]):
    values = np.array([d.value for d in data])
    ses_model = sm.tsa.SimpleExpSmoothing(values)
    ses_result = ses_model.fit()
    prediction = ses_result.forecast(3)
    return {"prediction": prediction.tolist()}

@app.post("/predict_hwes")
async def predict_hwes(data: List[TimeSeriesData]):
    values = np.array([d.value for d in data])
    hwes_model = sm.tsa.ExponentialSmoothing(values, seasonal_periods=4, trend='add', seasonal='add')
    hwes_result = hwes_model.fit()
    prediction = hwes_result.forecast(3)
    return {"prediction": prediction.tolist()}
