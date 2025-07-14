import cv2
import numpy as np
# from picamera2 import Picamera2 # No longer directly using Picamera2 here
import pytesseract
from pytesseract import Output
from PIL import Image, ImageTk
import tkinter as tk
import time
import logging # For consistent logging with object detection script

# Import PiCameraStream
try:
    from rpi_vision.agent.capturev2 import PiCameraStream
except ImportError:
    logging.error("rpi_vision library not found. Please ensure it's installed.")
    logging.error("You might need to install it: pip install rpi_vision")
    sys.exit(1) # Exit if PiCameraStream is not available as it's a core dependency now

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO) # Set logging level

# Initialize PiCameraStream
# Choose a resolution that suits your OCR needs. 640x480 is good.
desired_resolution = (640, 480)
capture_manager = PiCameraStream(preview=False, resolution=desired_resolution)
capture_manager.start()
time.sleep(1) # Give camera a moment to warm up

# Tkinter GUI setup
window = tk.Tk()
window.title("Adaptive Threshold with OCR Overlay")
label = tk.Label(window)
label.pack()

# Global flag to control OCR execution
ocr_enabled = False

# List to store the last OCR results for drawing
ocr_result = []

def on_key_press(event):
    """
    Callback function executed when a key is pressed.
    If the 't' key is pressed, it sets the ocr_enabled flag to True.
    """
    global ocr_enabled
    if event.char == 't':
        ocr_enabled = True
        print("OCR enabled for next frame processing.")

def update_frame():
    """
    Captures a frame from the camera using PiCameraStream, processes it, and updates the display.
    OCR is performed only if the ocr_enabled flag is True.
    """
    global ocr_enabled, ocr_result

    # Capture a raw array frame from the camera using PiCameraStream
    # .read() updates the internal .frame property, which is a numpy array.
    frame_raw = capture_manager.read()
    
    if frame_raw is None: # Handle case where no frame is available yet
        window.after(10, update_frame)
        return

    start = time.time() # Start time for FPS calculation

    # PiCameraStream.frame is already a numpy array (RGB888 by default config)
    # Convert the frame to grayscale for processing
    gray = cv2.cvtColor(capture_manager.frame, cv2.COLOR_RGB2GRAY)

    # Apply adaptive thresholding to the grayscale image
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   11, 2)
    # Convert the thresholded image back to BGR for color drawing (OpenCV default)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # --- OCR Logic (now conditional on key press) ---
    if ocr_enabled:
        print("Performing OCR...")
        # Perform OCR on the thresholded image
        d = pytesseract.image_to_data(thresh, output_type=Output.DICT)
        
        # Clear previous OCR results before populating new ones
        ocr_result = []
        
        # Iterate through detected text and store confident results
        for i in range(len(d['text'])):
            # Check confidence level and ensure text is not empty
            if int(d['conf'][i]) > 60 and d['text'][i].strip() != "":
                ocr_result.append((d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i]))
        
        # Reset the flag so OCR runs only once per 't' key press
        ocr_enabled = False

    # --- Drawing OCR Results ---
    # Draw bounding boxes and text for the last OCR results (whether new or old)
    for (text, x, y, w, h) in ocr_result:
        # Draw a green rectangle around the detected text
        thresh_color = cv2.rectangle(thresh_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Put the detected text above the rectangle in red
        thresh_color = cv2.putText(thresh_color, text, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- Display in Tkinter ---
    # Convert the OpenCV image (BGR) to RGB for PIL
    display_frame = cv2.cvtColor(thresh_color, cv2.COLOR_BGR2RGB)
    # Convert the numpy array image to a PIL Image
    img = Image.fromarray(display_frame)
    # Convert the PIL Image to a Tkinter PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    
    # Update the Tkinter label with the new image
    label.imgtk = imgtk
    label.configure(image=imgtk)
    
    # Schedule the next frame update after 10 milliseconds
    window.after(10, update_frame)

    end = time.time() # End time for FPS calculation

    seconds = end - start
    print (f"Time taken : {seconds:.3f} seconds")
    
    # Calculate and print frames per second
    fps  = 1 / seconds
    print(f"Estimated frames per second : {fps:.1f}")

# Function to be called when the window is closed
def on_closing():
    print("Stopping camera stream and closing application...")
    if capture_manager:
        capture_manager.stop() # Stop the PiCameraStream
    window.destroy()
    sys.exit(0) # Ensure the script exits cleanly

# Bind the '<Key>' event to the on_key_press function.
window.bind('<Key>', on_key_press)
# Bind the window close event to the on_closing function
window.protocol("WM_DELETE_WINDOW", on_closing)


# Start the frame update loop
update_frame()
# Start the Tkinter event loop
window.mainloop()
