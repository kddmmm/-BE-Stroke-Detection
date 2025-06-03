# app/utils/json_logger.py

import json
from datetime import datetime

def write_result(face_result, arm_result, voice_result, final_result, filename="result.json"):
    result = {
        "timestamp": datetime.now().isoformat(),
        "face": face_result,
        "arm": arm_result,
        "voice": voice_result,
        "final": final_result
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"[JSON 저장] {filename} 에 결과 저장됨")
