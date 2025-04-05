from ultralytics import YOLO
import cv2
import socket
import time

# Load your trained model (should be trained for rock, paper, scissors detection)
model = YOLO('path_to_your_rps_model.pt')  # Replace with path to your rock-paper-scissors trained model

# Define classes for rock, paper, scissors
class_names = ['rock', 'paper', 'scissors']  # Make sure these match your model's classes

# Set up UDP socket for communication with ESP32
esp32_ip = '192.168.1.100'  # Replace with your ESP32's IP address
esp32_port = 8888
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Variables to track detection and avoid spamming the ESP32
last_detection = None
last_sent_time = 0
send_interval = 1  # seconds between messages to avoid flooding

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Detection started. Press 'q' to quit.")

# Loop to capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Perform object detection on the frame
    results = model(frame)
    
    # Process the results to find rock, paper, scissors gestures
    current_detection = None
    highest_conf = 0.5  # Minimum confidence threshold
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            # If we found a gesture with higher confidence than previous ones
            if conf > highest_conf and cls_id < len(class_names):
                highest_conf = conf
                current_detection = class_names[cls_id]
    
    # If we have a detection and it's time to send another update
    current_time = time.time()
    if current_detection and (current_detection != last_detection or 
                             (current_time - last_sent_time) > send_interval):
        # Send the detection to ESP32
        message = current_detection.encode('utf-8')
        try:
            sock.sendto(message, (esp32_ip, esp32_port))
            print(f"Sent {current_detection} to ESP32")
            last_sent_time = current_time
            last_detection = current_detection
        except Exception as e:
            print(f"Error sending to ESP32: {e}")
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()  # Plot the detection results on the frame
    
    # Add text showing current detection
    if current_detection:
        cv2.putText(annotated_frame, f"Detected: {current_detection}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the annotated frame
    cv2.imshow("Rock Paper Scissors Detection", annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

