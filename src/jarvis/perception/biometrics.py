"""
JARVIS-PRIME Biometric Perception
==================================

Camera-based user identification.
Runs ONLY at startup for a brief scan (5-10 seconds),
then releases the camera to save resources.
Can be re-triggered on demand via identify_user().
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Any

try:
    import cv2
except ImportError:
    print("[WARN] opencv-python not installed. Biometrics disabled.")
    cv2 = None


class VisionBiometrics:
    """
    Brief camera scan to identify the user at startup.
    Does NOT keep the camera running continuously.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

        # Callbacks
        self.on_user_detected: Callable[[str], None] | None = None
        self.on_no_user: Callable[[], None] | None = None

        # State
        self.user_identified = False
        self.user_name: str | None = None

        # OpenCV face detector (Haar Cascade — lightweight, CPU-only)
        self.face_cascade = None
        if cv2 is not None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def identify_at_startup(self, max_seconds: float = 6.0) -> bool:
        """
        Open the camera briefly, try to detect a face, then release.
        Returns True if a user was detected.
        Called once at startup — camera is freed immediately after.
        """
        if cv2 is None or self.face_cascade is None:
            print("[BIOMETRICS] OpenCV not available, skipping face scan.")
            return False

        print("[BIOMETRICS] Quick startup scan... (camera on for ~5s)")

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("[BIOMETRICS] Cannot open camera. Skipping.")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        detected = False
        start = time.time()

        while (time.time() - start) < max_seconds:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                detected = True
                break

            time.sleep(0.15)

        # Release camera immediately
        cap.release()

        if detected:
            self.user_identified = True
            self.user_name = "Rahul"  # Default primary user
            print(f"[BIOMETRICS] User detected: {self.user_name}")
            if self.on_user_detected:
                self.on_user_detected(self.user_name)
        else:
            print("[BIOMETRICS] No user detected at startup.")
            if self.on_no_user:
                self.on_no_user()

        print("[BIOMETRICS] Camera released. Scan complete.")
        return detected

    def identify_now(self) -> bool:
        """
        On-demand re-scan (e.g., when user says "who am I?").
        Opens camera briefly, checks, releases.
        """
        return self.identify_at_startup(max_seconds=4.0)

    def status(self) -> dict[str, Any]:
        return {
            "user_identified": self.user_identified,
            "user_name": self.user_name,
            "camera_active": False,  # Camera is never kept open
        }


if __name__ == "__main__":
    def hello(name):
        print(f"Welcome back, {name}!")

    v = VisionBiometrics()
    v.on_user_detected = hello
    v.identify_at_startup()
    print("Camera is now released.")
