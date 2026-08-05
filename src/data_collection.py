import os
import cv2
import time
from src.utils import load_face_cascade

def collect_data(config, logger):
    """
    Launches an interactive OpenCV window to capture face images from the webcam.
    
    Requirements to implement:
    1. Open webcam feed (cv2.VideoCapture) using camera index from config.
    2. Load face detector to locate faces in the frame.
    3. Loop over incoming frames:
       - Detect faces in the current frame.
       - Highlight faces with a bounding box overlay.
       - Render HUD showing:
         * Selected emotion class
         * Key mapping instructions (e.g., keys 1-5 to switch classes)
         * Current image count for each emotion class
         * Auto-capture status (enabled/disabled)
       - Display the frame.
    4. Capture Trigger (manual: SPACE, or automatic time-interval toggle: 'c'):
       - Extract the bounding box crop of the primary detected face.
       - Save the face crop to `data/raw/{emotion_class}/face_{timestamp_or_index}.jpg`.
       - Update class counts dynamically.
    5. Key handlers:
       - '1' to '5': Switch target emotion class (Happy, Sad, Angry, Neutral, Surprise).
       - 'SPACE': Capture single crop.
       - 'c': Toggle auto-capture mode (e.g., save crop every N frames).
       - 'q': Exit data collection loop and release webcam resource.
       
    Args:
        config (dict): Configuration dictionary.
        logger (logging.Logger): Log helper.
    """
    logger.info("Initializing webcam data collection...")
    # TODO: Implement webcam loop, face cropping, key bindings, and frame overlays.
    raise NotImplementedError("Implement collect_data in src/data_collection.py")
