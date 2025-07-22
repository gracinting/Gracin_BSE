import os
import sys
import time
import binascii
import ctypes
import traceback
import subprocess
from contextlib import contextmanager
import threading

import requests
import openai
import speech_recognition as sr
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from picamera2 import Picamera2

# ----- Force UTF-8 output in console -----
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ----- Suppress those ALSA / Jack messages -----
try:
    snd = ctypes.cdll.LoadLibrary("libasound.so")
    snd.snd_lib_error_set_handler(None)
except OSError:
    print("Warning: libasound.so not found; ALSA suppression may be incomplete.")

@contextmanager
def suppress_stderr():
    orig_fd = sys.stderr.fileno()
    dup_fd = os.dup(orig_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, orig_fd)
    try:
        yield
    finally:
        os.dup2(dup_fd, orig_fd)
        os.close(dup_fd)
        os.close(devnull)

# ——— Configuration —————————————————————————————————————————————
OPENAI_API_KEY = "sk-proj-..."
VISION_MODEL   = "gpt-4-turbo"
VISION_PROMPT  = "Can you tell me what I'm looking at. You are an assistant, please keep the answers to one line."
IMAGE_PATH     = "/home/gracin/voice_assistant/assistant/images/capture.jpg"

openai.api_key = OPENAI_API_KEY

ESPEAK_VOICE     = "en-uk+f3"
ESPEAK_SPEED     = "140"
ESPEAK_PITCH     = "70"
ESPEAK_AMPLITUDE = "150"

# ——— Camera Globals —————————————————————————————————————————————
global_cam = None
global_frame = None

def start_live_camera():
    global global_cam, global_frame
    global_cam = Picamera2()
    config = global_cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    global_cam.configure(config)
    global_cam.start()

    def update_frame():
        global global_frame
        while True:
            global_frame = global_cam.capture_array()
            cv2.imshow("Live Camera Feed", global_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    t = threading.Thread(target=update_frame, daemon=True)
    t.start()

def capture_image(path: str) -> str:
    global global_frame
    if global_frame is not None:
        cv2.imwrite(path, cv2.cvtColor(global_frame, cv2.COLOR_RGB2BGR))
        print(f"[+] Image saved to {path}")
    else:
        print("[!] No frame available from camera.")
    return path

def speak_festival(text: str):
    subprocess.run([
        "espeak",
        "-v", ESPEAK_VOICE,
        "-s", ESPEAK_SPEED,
        "-p", ESPEAK_PITCH,
        "-a", ESPEAK_AMPLITUDE,
        text
    ], check=True)


def encode_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return binascii.b2a_base64(f.read()).decode().strip()

def send_to_openai_vision(image_path: str, prompt: str):
    b64 = encode_to_base64(image_path)
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ]
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type":  "application/json"
    }
    print("[+] Sending image to OpenAI Vision…")
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json=payload, headers=headers
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip()
    print("Vision API response:", answer)
    speak_festival(answer)

def object_detection():
    print("Capturing image for object detection…")
    img = capture_image(IMAGE_PATH)
    send_to_openai_vision(img, VISION_PROMPT)

def handle_ocr():
    print("Capturing image for OCR…")
    img_path = capture_image(IMAGE_PATH)
    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to load image.")
        speak_festival("Sorry, I couldn't read the image.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    print("🔍 Running OCR…")
    data = pytesseract.image_to_data(thresh, output_type=Output.DICT)
    words = []
    for i, txt in enumerate(data["text"]):
        txt = txt.strip()
        try:
            conf = int(data["conf"][i])
        except:
            continue
        if txt and conf > 60:
            x, y, w, h = (data[k][i] for k in ("left", "top", "width", "height"))
            words.append((txt, x, y, w, h))

    if not words:
        print("No confident text found.")
        speak_festival("I couldn't detect any clear text.")
        return

    words.sort(key=lambda w: (w[2], w[1]))
    lines = []
    current, last_y = [], None
    tol = 10
    for word in words:
        txt, x, y, w, h = word
        if last_y is None or abs(y - last_y) <= tol:
            current.append(word)
        else:
            lines.append(current)
            current = [word]
        last_y = y
    if current:
        lines.append(current)

    for line in lines:
        line = sorted(line, key=lambda w: w[1])
        line_text = " ".join(w[0] for w in line)
        print("Line:", line_text)
        speak_festival(line_text)

def main():
    start_live_camera()

    WAKE_WORD    = "aurora"
    PAUSE_PHRASE = "thank you"
    EXIT_PHRASES = {"exit", "quit", "stop", "goodbye"}

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 150

    with suppress_stderr(), sr.Microphone(device_index=3, sample_rate=44100, chunk_size=2048) as src:
        print("Calibrating ambient noise for 3 second…")
        recognizer.adjust_for_ambient_noise(src, duration=2)

    print(f"Calibrated energy threshold: {recognizer.energy_threshold}\n")
    print(f"Say '{WAKE_WORD}' to wake me, or any EXIT word to quit.\n")

    listening_mode = False

    while True:
        try:
            with suppress_stderr(), sr.Microphone(device_index=3) as src:
                if not listening_mode:
                    print(f"Listening for wake word '{WAKE_WORD}'…")
                    audio = recognizer.listen(src, phrase_time_limit=3)
                else:
                    print("Listening for your command…")
                    audio = recognizer.listen(src)

            try:
                phrase = recognizer.recognize_google(audio).lower()
                print(f"Recognized: {phrase}")
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Speech API error: {e}")
                continue

            if any(exit_p in phrase for exit_p in EXIT_PHRASES):
                print("Goodbye!")
                break

            if not listening_mode:
                if WAKE_WORD in phrase or "test" in phrase:
                    listening_mode = True
                    print("Wake word detected. I'm now listening freely.")
                    speak_festival("Yes?")
                continue

            if PAUSE_PHRASE in phrase:
                listening_mode = False
                print("Heard 'thank you'. Going quiet until called again.")
                speak_festival("You're welcome.")
                continue

            if "what does this say" in phrase or "read this" in phrase or "ocr" in phrase:
                handle_ocr()
                continue

            if "what is this" in phrase or "object detection" in phrase:
                object_detection()
                continue

            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are Aurora, a helpful assistant, reply in two lines."},
                        {"role": "user",   "content": phrase}
                    ]
                )
                reply = resp.choices[0].message.content.strip()
                print(f"\nAurora says:\n{reply}\n")
                speak_festival(reply)
            except openai.OpenAIError as e:
                print(f"[OpenAI API Error] {e}\n")
                continue

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as err:
            print(f"[Unexpected Error] {type(err).__name__}: {err}\nRestarting…\n")
            traceback.print_exc()
            continue

    if global_cam:
        global_cam.stop()
        global_cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
