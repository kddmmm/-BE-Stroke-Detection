# app/utils/webcam.py

import cv2
import threading
import time

class WebcamManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    @classmethod
    def instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = WebcamManager()
                cls._instance.start()
            return cls._instance

    def start(self):
        if not self.running:
            self.running = True
            self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.03)

    def get_rgb_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            frame = cv2.flip(self.latest_frame, 1)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()
        cv2.destroyAllWindows()

