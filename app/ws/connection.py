from app.utils.webcam import WebcamManager
from app.detector.face import predict_face_asymmetry
from app.detector.arm import detect_arm
from app.detector.voice import detect_voice
from app.utils.json_logger import write_result
from fastapi import APIRouter, WebSocket
import json
import asyncio

websocket_router = APIRouter()

@websocket_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] 클라이언트 연결됨")

    try:
        while True:
            await asyncio.sleep(1)

            rgb_frame = WebcamManager.instance().get_rgb_frame()
            if rgb_frame is None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "no rgb frame"
                }))
                continue

            face_result_obj = predict_face_asymmetry(rgb_frame)
            await websocket.send_text(json.dumps({
                "type": "face",
                "value": face_result_obj["Result"],
                "rate": face_result_obj["rate"]
            }))

            if face_result_obj["Result"] in ["failed to detect face", "no front face"]:
                continue

            frame_count = 30
            valid_results = []
            valid_rates = []
            for _ in range(frame_count):
                frame = WebcamManager.instance().get_rgb_frame()
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                res = predict_face_asymmetry(frame)
                if res["Result"] in ["failed to detect face", "no front face"]:
                    await asyncio.sleep(0.1)
                    continue
                valid_results.append(1 if res["Result"] == "abnormal" else 0)
                valid_rates.append(res["rate"])
                await asyncio.sleep(0.1)

            if not valid_results:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "failed to detect face in 30 frames"
                }))
                continue

            normal_ratio = sum(valid_results) / len(valid_results)
            final_face_result = "abnormal" if normal_ratio >= 0.5 else "normal"

            await websocket.send_text(json.dumps({
                "type": "face-final",
                "value": final_face_result,
                "rate": round(normal_ratio, 4)
            }))

            if final_face_result == "normal":
                continue
            elif final_face_result == "abnormal":
                await asyncio.sleep(5)
                # 팔 감지
                while True:
                    arm_result_obj = detect_arm()
                    if arm_result_obj["result"] == "no arm frame":
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "failed to detect arm"
                        }))
                        continue
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "arm",
                            "value": arm_result_obj["result"],
                            "rate": arm_result_obj["rate"]
                        }))
                        break
                await asyncio.sleep(5)
                # 음성 감지
                voice_result_obj = detect_voice()
                await websocket.send_text(json.dumps({
                    "type": "voice",
                    "value": voice_result_obj["value"],
                    "rate": voice_result_obj["rate"]
                }))
                final_value = 0.5 * arm_result_obj["rate"] + 0.3 * normal_ratio + 0.2 * voice_result_obj["rate"]
                if final_value < 0.5:
                    await websocket.send_text(json.dumps({
                        "type": "final",
                        "value": "normal",
                        "rate": final_value
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "final",
                        "value": "abnormal",
                        "rate": final_value
                    }))
                write_result(
                    {"Result": final_face_result, "rate": round(normal_ratio, 4)},
                    arm_result_obj,
                    voice_result_obj,
                    final_value
                )
                await asyncio.sleep(10)

    except Exception as e:
        print(f"[WebSocket 오류] {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"서버 내부 오류: {e}"
        }))
    finally:
        print("[WebSocket] 연결 종료됨")
        try:
            await websocket.close()
        except Exception:
            pass
