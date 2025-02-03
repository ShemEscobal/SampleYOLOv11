from ultralytics import YOLO

if __name__ == '__main__':

    # Create a model
    model = YOLO('yolo11n.pt')  # Start with pre-trained weights

    # Train the model using all available GPUs
    results = model.train(
        data=r'your path to data yaml file',  # Path to data.yaml file
        epochs=100,
        imgsz=640,
        batch=16,  # Adjust batch size 
        patience=50,
        device='0'  # Use all available GPUs
    )