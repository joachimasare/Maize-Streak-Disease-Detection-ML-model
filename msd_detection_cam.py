import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'maize_leaf_disease_detection_model.h5' 
model = load_model(model_path)

def preprocess_image(image):
    #Preprocess the image to be compatible with the model.
    image = cv2.resize(image, (224, 224))  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize to [0,1]
    return image

def predict(image):
    #Predict if the leaf is healthy or infected.
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]

# Initialize USB webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('Maize Leaf Scanner', frame)

    # Press 's' to save and predict
    if cv2.waitKey(1) & 0xFF == ord('s'):
        prediction = predict(frame)
        if prediction > 0.5:
            print(f"Infected with probability: {prediction * 100:.2f}%")
        else:
            print(f"Healthy with probability: {(1 - prediction) * 100:.2f}%")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
