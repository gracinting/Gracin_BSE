# SPDX-FileCopyrightText: 2021 Limor Fried/ladyada for Adafruit Industries
# SPDX-FileCopyrightText: 2021 Melissa LeBlanc-Washington for Adafruit Industries
#
# SPDX-License-Identifier: MIT

import time
import logging
import argparse
import os
import subprocess
import sys
import numpy as np
import signal
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk, ImageOps

# Suppress PIL's INFO messages
logging.getLogger('PIL').setLevel(logging.WARNING)

CONFIDENCE_THRESHOLD = 0.5   # at what confidence level do we say we detected a thing
PERSISTANCE_THRESHOLD = 0.25  # what percentage of the time we have to have seen a thing

def dont_quit(signal_num, frame):
   """Signal handler to prevent quitting on SIGHUP."""
   print(f'Caught signal: {signal_num}')
signal.signal(signal.SIGHUP, dont_quit)

# App components (assuming these are installed or available)
try:
    from rpi_vision.agent.capturev2 import PiCameraStream
    from rpi_vision.models.mobilenet_v2 import MobileNetV2Base
except ImportError:
    logging.error("rpi_vision library not found. Please ensure it's installed.")
    logging.error("You might need to install it: pip install rpi_vision")
    sys.exit(1)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class VisionApp:
    def __init__(self, args):
        self.args = args
        self.root = tk.Tk()
        self.root.title("MobileNetV2 Vision App")

        self.root.geometry(f"{int(self.root.winfo_screenwidth() * 0.8)}x{int(self.root.winfo_screenheight() * 0.8)}")
        self.root.geometry("800x600")

        # Create a canvas to display the video feed and overlays
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.capture_manager = None
        self.model = None
        self.last_seen = [None] * 10
        self.last_spoken = None
        self.detection_enabled = False #  Initial state for detection

        self.init_fonts()
        self.load_splash_screen()

        self.root.bind("<Configure>", self.on_resize)
        self.root.bind("o", self.toggle_detection)
        self.root.bind("O", self.toggle_detection)

        self.root.update_idletasks() # Ensure window dimensions are updated before camera init

        self.init_camera_and_model()

        # Start the update loop
        self.update_frame()

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_fonts(self):
        """Initializes Tkinter fonts based on screen resolution."""
        # Scale font sizes relative to current canvas height for better responsiveness
        # We'll re-evaluate these on resize if needed, but for initial setup, use screen height
        canvas_height = self.root.winfo_screenheight() # Use screen height for initial font sizing
        self.smallfont = font.Font(family="Helvetica", size=int(canvas_height * 0.02))
        self.medfont = font.Font(family="Helvetica", size=int(canvas_height * 0.03))
        self.bigfont = font.Font(family="Helvetica", size=int(canvas_height * 0.04))
        # New: Font for detection status
        self.statusfont = font.Font(family="Helvetica", size=int(canvas_height * 0.025), weight="bold")

    def on_resize(self, event):
        """Handles window resizing to adjust font sizes and redraw content."""
        # Only update if the canvas size has actually changed
        if self.canvas.winfo_width() != self.current_canvas_width or \
           self.canvas.winfo_height() != self.current_canvas_height:
            self.current_canvas_width = self.canvas.winfo_width()
            self.current_canvas_height = self.canvas.winfo_height()
            self.init_fonts() # Re-initialize fonts based on new canvas size
            # No need to call update_frame here, it will be called by root.after

    def load_splash_screen(self):
        """Loads and displays a splash screen."""
        try:
            splash_path = os.path.join(os.path.dirname(sys.argv[0]), 'bchatsplash.bmp')
            if not os.path.exists(splash_path):
                logging.warning(f"Splash screen not found at {splash_path}. Skipping splash.")
                # Create a simple black background if splash is not found
                self.canvas.create_rectangle(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height(), fill="black")
                self.canvas.create_text(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                         text="Loading...", fill="white", font=self.bigfont)
                self.root.update()
                return

            splash_img = Image.open(splash_path)

            # Rotate the splash image if specified
            if self.args.rotation != 0:
                splash_img = splash_img.rotate(self.args.rotation, expand=True)

            # Scale the image to fit the smaller dimension of the current canvas, maintaining aspect ratio
            img_width, img_height = splash_img.size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            scale_factor = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            splash_img = splash_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            self.tk_splash_img = ImageTk.PhotoImage(splash_img)

            # Center the image on the canvas
            self.canvas.create_image(canvas_width / 2, canvas_height / 2,
                                     image=self.tk_splash_img, anchor=tk.CENTER)
            self.root.update() # Update the display to show splash screen
        except Exception as e:
            logging.error(f"Error loading or displaying splash screen: {e}")
            self.canvas.create_rectangle(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height(), fill="black")
            self.canvas.create_text(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                     text="Loading...", fill="white", font=self.bigfont)
            self.root.update()

    def init_camera_and_model(self):
        """Initializes the camera stream and the MobileNetV2 model."""
        try:
            # --- MODIFICATION START ---
            # Define your desired resolution here. Common smaller resolutions are:
            # (640, 480), (320, 240), (160, 120)
            # Choose a resolution that suits your needs.
            desired_resolution = (320, 240) # Example: VGA resolution

            self.capture_manager = PiCameraStream(preview=False, resolution=desired_resolution)
            # --- MODIFICATION END ---
            self.capture_manager.start()
            # Wait a bit for the camera to warm up and get the first frame
            time.sleep(2)

            # Ensure camera resolution is available
            if not self.capture_manager.resolution:
                raise RuntimeError("Failed to get camera resolution from PiCameraStream.")

            self.model = MobileNetV2Base(include_top=self.args.include_top)
            logging.info("Camera and model initialized successfully.")

            # Initialize current canvas dimensions for resize tracking
            self.current_canvas_width = self.canvas.winfo_width()
            self.current_canvas_height = self.canvas.winfo_height()

        except Exception as e:
            logging.error(f"Failed to initialize camera or model: {e}")
            self.on_closing() # Close the app if initialization fails

    def toggle_detection(self, event=None):
        """Toggles the object detection functionality on and off."""
        self.detection_enabled = not self.detection_enabled
        logging.info(f"Object detection is now {'ENABLED' if self.detection_enabled else 'DISABLED'}")
        if not self.detection_enabled:
            # Clear last seen objects and spoken text when detection is turned off
            self.last_seen = [None] * 10
            self.last_spoken = None

    def update_frame(self):
        """
        Reads a frame from the camera, performs inference (if enabled), and updates the Tkinter canvas.
        This function is called periodically.
        """
        if self.capture_manager is None or self.capture_manager.stopped:
            self.root.after(100, self.update_frame) # Try again soon if not ready
            return

        frame_raw = self.capture_manager.read()
        if frame_raw is None:
            self.root.after(10, self.update_frame) # Try again soon
            return

        # Get current canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Clear previous drawings on the canvas
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="black")

        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(self.capture_manager.frame)

        # Apply rotation if specified
        if self.args.rotation != 0:
            pil_img = pil_img.rotate(self.args.rotation, expand=True)

        # Resize the PIL image to fit the current canvas, maintaining aspect ratio
        img_width, img_height = pil_img.size

        # Calculate scale factor to fit the image within the canvas dimensions
        if img_width > 0 and img_height > 0: # Avoid division by zero
            scale_factor = min(canvas_width / img_width, canvas_height / img_height)
        else:
            scale_factor = 1.0 # Default if image has no dimensions

        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        # Ensure dimensions are at least 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert PIL Image to Tkinter PhotoImage
        self.tk_img = ImageTk.PhotoImage(pil_img)

        # Center the image on the canvas
        self.canvas.create_image(canvas_width / 2, canvas_height / 2,
                                 image=self.tk_img, anchor=tk.CENTER)

        timestamp = time.monotonic()

        # Display detection status
        status_text = "Detection: " + ("ON" if self.detection_enabled else "OFF")
        status_color = "green" if self.detection_enabled else "red"
        self.canvas.create_text(10, 10, text=status_text, fill=status_color,
                                font=self.statusfont, anchor=tk.NW)

        if self.detection_enabled and self.model: # Only run detection if enabled and model is initialized
            if self.args.tflite:
                prediction = self.model.tflite_predict(frame_raw)[0]
            else:
                prediction = self.model.predict(frame_raw)[0]
            delta = time.monotonic() - timestamp
            logging.info("%s inference took %d ms, %0.1f FPS" % ("TFLite" if self.args.tflite else "TF", delta * 1000, 1 / delta))
            # print(self.last_seen) # For debugging

            # Add FPS & temp on top corner of image
            fpstext = "%0.1f FPS" % (1/delta,)
            self.canvas.create_text(canvas_width - 10, 10, text=fpstext, fill="red",
                                    font=self.smallfont, anchor=tk.NE)
            try:
                temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
                temptext = f"{temp:.0f}°C"
                self.canvas.create_text(canvas_width - 10, 30, text=temptext, fill="red",
                                        font=self.smallfont, anchor=tk.NE)
            except OSError:
                pass # Not on a Raspberry Pi or thermal zone not found

            detected_something = False
            for p in prediction:
                label, name, conf = p
                if conf > CONFIDENCE_THRESHOLD:
                    detected_something = True
                    logging.info(f"Detected: {name} (Confidence: {conf:.2f})")

                    persistant_obj = False
                    self.last_seen.append(name)
                    self.last_seen.pop(0)

                    inferred_times = self.last_seen.count(name)
                    if inferred_times / len(self.last_seen) > PERSISTANCE_THRESHOLD:
                        persistant_obj = True

                    detecttext = name.replace("_", " ")
                    detecttext_font = self.smallfont # Default
                    # Find the largest font that fits
                    for f in (self.bigfont, self.medfont, self.smallfont):
                        text_width = f.measure(detecttext)
                        if text_width < canvas_width * 0.9: # Allow some margin
                            detecttext_font = f
                            break

                    detecttext_color = "green" if persistant_obj else "white"
                    # Position at the bottom center of the screen
                    text_y_position = canvas_height - detecttext_font.metrics("linespace") - 10
                    self.canvas.create_text(canvas_width // 2, text_y_position,
                                            text=detecttext, fill=detecttext_color,
                                            font=detecttext_font, anchor=tk.S)

                    if persistant_obj and self.last_spoken != detecttext:
                        try:
                            subprocess.Popen(f"echo '{detecttext}' | festival --tts &", shell=True)
                            self.last_spoken = detecttext
                        except FileNotFoundError:
                            logging.warning("Festival TTS not found. Please install it if you want speech output.")
                        except Exception as e:
                            logging.error(f"Error calling festival TTS: {e}")
                    break # Only show the highest confidence detection

            if not detected_something:
                self.last_seen.append(None)
                self.last_seen.pop(0)
                if all(item is None for item in self.last_seen):
                    self.last_spoken = None
        elif not self.detection_enabled:
            # Optionally display a message that detection is off
            self.canvas.create_text(canvas_width // 2, canvas_height // 2,
                                     text="Object Detection OFF (Press 'O' to enable)",
                                     fill="white", font=self.medfont, anchor=tk.CENTER)
            # Clear any persistent detection data if detection is off
            self.last_seen = [None] * 10
            self.last_spoken = None

        # Schedule the next update
        self.root.after(10, self.update_frame) # Update every 10 milliseconds (approx 100 FPS)

    def on_closing(self):
        """Handles the window closing event, stopping the camera stream."""
        logging.info("Closing application...")
        if self.capture_manager:
            self.capture_manager.stop()
        self.root.destroy()
        sys.exit(0) # Ensure the script exits cleanly

    def run(self):
        """Starts the Tkinter event loop."""
        self.root.mainloop()

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool,
                        dest='include_top', default=True,
                        help='Include fully-connected layer at the top of the network.')

    parser.add_argument('--tflite',
                        dest='tflite', action='store_true', default=False,
                        help='Convert base model to TFLite FlatBuffer, then load model into TFLite Python Interpreter')

    parser.add_argument('--rotation', type=int, choices=[0, 90, 180, 270],
                        dest='rotation', action='store', default=0,
                        help='Rotate everything on the display by this amount (0, 90, 180, 270 degrees).')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    app = VisionApp(args)
    try:
        app.run()
    except KeyboardInterrupt:
        app.on_closing()
    except Exception as e:
        logging.error(f"An unhandled error occurred: {e}")
        if app.capture_manager:
            app.capture_manager.stop()
        sys.exit(1)
