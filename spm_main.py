from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import statsmodels.api as sm
from collections import defaultdict

app = FastAPI()

# 시계열 데이터에 대한 데이터 모델 정의
class TimeSeriesData(BaseModel):
    id: str
    date: str
    value: float

# 데이터 포인트가 충분한지 확인하는 함수
def check_minimum_data_points(data, min_points):
    for id_, id_data in data.items():
        if len(id_data) < min_points:
            return False
    return True

# 예측할 다음 날짜들을 생성하는 함수
def get_next_dates(last_date, num_dates):
    last_year, last_month = int(last_date[:4]), int(last_date[4:])
    next_dates = []
    for _ in range(num_dates):
        last_month += 1
        if last_month > 12:
            last_month = 1
            last_year += 1
        next_dates.append(f"{last_year:04d}{last_month:02d}")
    return next_dates

# 단순 지수 평활법(Simple Exponential Smoothing)을 사용하여 예측하는 함수
@app.post("/predict_ses")
async def predict_ses(data: List[TimeSeriesData]):
    id_to_data = defaultdict(list)  # id별로 데이터를 그룹화할 딕셔너리
    for d in data:
        id_to_data[d.id].append(d)  # 데이터를 id별로 분류

    predictions = {}  # 각 id에 대한 예측값을 저장할 딕셔너리
    for id_, id_data in id_to_data.items():
        values = np.array([d.value for d in id_data])  # 데이터에서 값들만 추출
        ses_model = sm.tsa.SimpleExpSmoothing(values)  # 단순 지수 평활 모델 생성
        ses_result = ses_model.fit()  # 모델 학습
        prediction = ses_result.forecast(3)  # 3개월치 예측값 생성
        prediction = np.round(prediction, 1)  # 예측값을 소수점 첫째 자리에서 반올림
        next_dates = get_next_dates(id_data[-1].date, 3)  # 예측할 다음 날짜들 생성
        predictions[id_] = list(zip(next_dates, prediction.tolist()))  # 예측 결과 저장

    return {"predictions": predictions}

# 홀트-윈터스(Holt-Winters)를 사용하여 예측하는 함수
@app.post("/predict_hwes")
async def predict_hwes(data: List[TimeSeriesData]):
    id_to_data = defaultdict(list)  # id별로 데이터를 그룹화할 딕셔너리
    for d in data:
        id_to_data[d.id].append(d)  # 데이터를 id별로 분류

    min_data_points = 8  # 최소 관측치 수
    # 데이터 포인트가 충분한지 확인
    if not check_minimum_data_points(id_to_data, min_data_points):
        raise HTTPException(status_code=400, detail="각 ID별로 최소한 8개의 관측치가 필요합니다.")

    predictions = {}  # 각 id에 대한 예측값을 저장할 딕셔너리
    for id_, id_data in id_to_data.items():
        values = np.array([d.value for d in id_data])  # 데이터에서 값들만 추출
        hwes_model = sm.tsa.ExponentialSmoothing(values, seasonal_periods=4, trend='add', seasonal='add')  # 홀트-윈터스 모델 생성
        hwes_model = sm.tsa.ExponentialSmoothing(values, seasonal_periods=4, trend='add',
                                                 seasonal='add')  # 홀트-윈터스 모델 생성
        hwes_result = hwes_model.fit()  # 모델 학습
        prediction = hwes_result.forecast(3)  # 3개월치 예측값 생성
        prediction = np.round(prediction, 1)  # 예측값을 소수점 첫째 자리에서 반올림
        next_dates = get_next_dates(id_data[-1].date, 3)  # 예측할 다음 날짜들 생성
        predictions[id_] = list(zip(next_dates, prediction.tolist()))  # 예측 결과 저장

    return {"predictions": predictions}

