import torch.nn as nn
import torch
import pickle
import pandas as pd
from app.extract.extract_voice_features import extract_features

class SpeechDisorderDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

MODEL_PATH = "app/models/voice/voice_model.pt"
SCALER_PATH = "app/models/voice/voice_scaler.pkl"

def load_model_and_scaler():
    try:
        model = SpeechDisorderDetector()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
    except Exception as e:
        print(f"[MODEL LOAD ERROR] {e}")
        model = None

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"[SCALER LOAD ERROR] {e}")
        scaler = None

    return model, scaler

def detect_voice():
    try:
        from app.utils.recorder import record_audio
        model, scaler = load_model_and_scaler()

        if model is None or scaler is None:
            print("[DETECT_VOICE ERROR] 모델 또는 스케일러가 None입니다.")
            return "unknown"

        record_audio("temp_audio.wav")
        features = extract_features("temp_audio.wav")

        df = pd.DataFrame([features], columns=scaler.feature_names_in_)
        X_scaled = scaler.transform(df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = model(X_tensor)
            prob = torch.sigmoid(logits).item()
            prediction = "abnormal" if prob >= 0.5 else "normal"

        return {
            "value": prediction,
            "rate": prob
        }
    except Exception as e:
        print(f"[DETECT_VOICE ERROR] 예측 중 오류: {e}")
        return "unknown"
