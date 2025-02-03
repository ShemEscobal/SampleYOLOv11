from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('path_to_your_trained_model.pt')  # Replace with the path to your trained model

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()  # Plot the detection results on the frame

    # Display the annotated frame
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()