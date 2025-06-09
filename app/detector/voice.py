import torch.nn as nn
import torch
import joblib
from app.extract.extract_voice_features import extract_features

class SpeechDisorderDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

MODEL_PATH = "app/models/voice/my_model.pt"
SCALER_PATH = "app/models/voice/my_scaler.pkl"

def load_model_and_scaler():

    model = SpeechDisorderDetector()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"[MODEL LOAD ERROR] {e}")
        return None, None
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"[SCALER LOAD ERROR] {e}")
        return None, None
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

        if features is None or len(features) != 8:
            print("[DETECT_VOICE ERROR] 특성 추출 오류 또는 개수 불일치")
            return "unknown"

        X_scaled = scaler.transform([features])
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

if __name__ == "__main__":
    result = detect_voice()
    print(result)
