import pytest
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8001/predict"
CSV_PATH = "test_data.csv"

@pytest.fixture
def load_test_data():
    df = pd.read_csv(CSV_PATH)
    payload = []
    for _, row in df.iterrows():
        payload.append({
            "camera_regno": row["regno_recognize"],
            "nn_regno": row["afts_regno_ai"],
            "camera_score": float(row["recognition_accuracy"]),
            "nn_score": float(row["afts_regno_ai_score"]),
            "nn_sym_scores": row["afts_regno_ai_char_scores"],
            "nn_len_scores": row["afts_regno_ai_length_scores"],
            "camera_type": str(row["camera_type"]),
            "camera_class": str(row["camera_class"]),
            "time_check": row["time_check"],
            "direction": str(row["direction"])
        })
    return payload

def test_predict_response_schema(load_test_data):
    response = requests.post(API_URL, json=load_test_data)
    assert response.status_code == 200
    body = response.json()
    for result in body["results"]:
        if isinstance(result, list):
            assert len(result) == 2
            assert all(isinstance(p, float) for p in result)
