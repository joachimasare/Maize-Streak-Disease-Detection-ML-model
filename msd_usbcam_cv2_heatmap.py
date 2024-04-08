import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

# Initialize the USB webcam
camera = cv2.VideoCapture(0)

# Directory to save images
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)
image_count = 0

try:
    for _ in range(10):  # Capture 10 images for example
        ret, img = camera.read()
        if not ret:
            break
        filename = os.path.join(save_dir, f"image_{image_count}.jpg")
        cv2.imwrite(filename, img)
        image_count += 1
        time.sleep(5)  # Wait for 5 seconds before capturing the next image

finally:
    camera.release()

# Process images and store probabilities
probabilities = []
image_paths = [os.path.join(save_dir, img) for img in os.listdir(save_dir)]
for img_path in image_paths:
    img = Image.open(img_path)
    img_array = preprocess_image(np.array(img))
    probability = model.predict(img_array)[0][0]
    probabilities.append(probability)

# Create collage
grid_size = (5, 5)  # Adjust grid size based on the number of images
collage_width = grid_size[0] * 224
collage_height = grid_size[1] * 224
collage = Image.new('RGB', (collage_width, collage_height))

x_offset = 0
y_offset = 0
for img_path in image_paths:
    img = Image.open(img_path)
    img = img.resize((224, 224))
    collage.paste(img, (x_offset, y_offset))
    x_offset += 224
    if x_offset >= collage_width:
        x_offset = 0
        y_offset += 224

# Generate heatmap
heatmap = np.zeros((grid_size[1], grid_size[0]))
for i, prob in enumerate(probabilities):
    row = i // grid_size[0]
    col = i % grid_size[0]
    heatmap[row, col] = prob

# Overlay heatmap
plt.imshow(collage)
plt.imshow(heatmap, cmap='hot', alpha=0.5, extent=[0, collage_width, 0, collage_height])
plt.colorbar()
plt.show()
