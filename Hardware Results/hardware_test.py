import cv2
import numpy as np
import tensorflow as tf
import time
import psutil  # For CPU & RAM usage

# Load Keras model (.h5)
model_path = "/home/nvidia/Desktop/MobileNetV1.h5"
model = tf.keras.models.load_model(model_path)

# Define input shape
IMG_SIZE = (224, 224)  # Change based on model input size

# Define class labels
class_labels = [
    "Acne and Rosacea",
    "Chickenpox",
    "Cowpox",
    "HFMD",
    "Healthy",
    "Measles",
    "Monkeypox"
]

# Set the mobile camera URL (Change this to your phone’s actual IP)
url = "http://192.168.1.100:8080/video"

cap = cv2.VideoCapture(url)

# Set FPS limit to reduce lag
TARGET_FPS = 10
frame_time = 1.0 / TARGET_FPS

# Performance tracking
frame_count = 0
total_inference_time = 0
start_time = time.time()

while True:
    frame_start_time = time.time()  # Start frame timer

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Resize and preprocess the full frame
    img = cv2.resize(frame, IMG_SIZE)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)

    # Measure inference time
    inference_start = time.time()
    prediction = model.predict(img)
    inference_time = time.time() - inference_start  # Time taken for prediction

    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_label = class_labels[class_index]

    # Display result on the frame (with black text & white outline for visibility)
    result_text = f"{predicted_label}: {confidence:.2f}%"
    text_position = (50, 50)

    cv2.putText(frame, result_text, text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)  # White outline
    cv2.putText(frame, result_text, text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)  # Black main text

    # Measure FPS
    frame_end_time = time.time()
    frame_duration = frame_end_time - frame_start_time
    fps = 1.0 / frame_duration if frame_duration > 0 else 0

    # Display FPS, Inference Time, and Hardware Stats
    stats_text = f"FPS: {fps:.2f} | Inference: {inference_time:.4f}s | CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%"
    stats_position = (50, 100)

    cv2.putText(frame, stats_text, stats_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)  # White outline
    cv2.putText(frame, stats_text, stats_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Black main text

    # Show the frame
    cv2.imshow("Skin Disease Classification", frame)

    # Update total inference time
    total_inference_time += inference_time
    frame_count += 1

    # Maintain FPS limit
    elapsed_time = time.time() - frame_start_time
    if elapsed_time < frame_time:
        time.sleep(frame_time - elapsed_time)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

# Calculate average inference time & FPS
total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0
avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0

print(f"Total Frames: {frame_count}")
print(f"Total Execution Time: {total_time:.2f} sec")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Average Inference Time: {avg_inference_time:.4f} sec")
print(f"Final CPU Usage: {psutil.cpu_percent()}%")
print(f"Final RAM Usage: {psutil.virtual_memory().percent}%")

cap.release()
cv2.destroyAllWindows()
