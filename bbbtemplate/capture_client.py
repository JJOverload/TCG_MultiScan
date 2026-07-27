"""
capture_client.py

Runs ON the BeagleBone Black. Captures a photo from the attached camera
and POSTs it to the scanning backend running on the more powerful machine.
Deliberately has no OpenCV/OCR/vocab dependencies beyond frame capture —
the BBB's job is just "take the picture and send it".
"""

from __future__ import annotations

import argparse
import time

import cv2 as cv
import requests


def capture_and_send(backend_url: str, camera_index: int) -> dict:
    cam = cv.VideoCapture(camera_index)
    try:
        ok, frame = cam.read()
    finally:
        cam.release()

    if not ok:
        raise RuntimeError(f"Failed to read from camera index {camera_index}")

    ok, jpeg = cv.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode captured frame as JPEG")

    response = requests.post(
        f"{backend_url}/api/scans",
        files={"image": ("capture.jpg", jpeg.tobytes(), "image/jpeg")},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a card photo and send it to the scanner backend.")
    parser.add_argument(
        "--backend-url", required=True,
        help="e.g. http://100.x.x.x:8000 (Tailscale IP) or http://scanner.local:8000",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument(
        "--interval", type=float, default=0,
        help="Seconds between captures. 0 (default) takes a single photo and exits.",
    )
    args = parser.parse_args()

    if args.interval <= 0:
        print(capture_and_send(args.backend_url, args.camera))
        return

    while True:
        try:
            print(capture_and_send(args.backend_url, args.camera))
        except Exception as exc:  # keep the capture loop alive across transient failures
            print(f"Capture failed: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
