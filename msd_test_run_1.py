import tensorflow as tf
import numpy as np
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model('path_to_your_model/maize_leaf_disease_detection_model.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize to match model's expected input
    img = img.astype('float32') / 255  # Scale pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict
def predict(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    return prediction[0][0]

# Initialize the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
raw_capture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(0.1)

try:
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        # Grab the raw NumPy array representing the image
        img = frame.array

        # Predict infection
        probability = predict(img)
        print(f"Probability of being infected: {probability * 100:.2f}%")

        # Clear the stream in preparation for the next frame
        raw_capture.truncate(0)

        # Display the image - this is optional and can be removed if not needed
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Wait for 5 seconds before capturing the next image
        time.sleep(5)

finally:
    cv2.destroyAllWindows()
