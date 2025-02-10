# SAMPLE YOLOv11 PRACTICE 

by Mechatronics and Robotics Society of the Philippines (MRSP) - Cavite Chapter

Use this repository to practice setting-up YOLOv11 using custom dataset downloaded from Roboflow

# OUTLINE
- Setting Up YOLOv11
- Preparing a Custom Dataset
- YAML Configuration for Custom Dataset
- Training YOLOv11 with Custom Dataset
- Running Inference with the Custom Model

# SETTING-UP OF YOLOv11
- Open VS Code and create a designated folder
- Create a python file name “YOLOtrain.py”. Or you can also use the file in this repository.
- Create virtual environment. Go to search box and select Show and Run Commands, select “Python: Create Environment” and select the available Python version that displays.
Note: For those who don’t have python installed on their pc install it from “python.org/downloads”
- Open the “Terminal > New Terminal”. You can locate it on the upper side.
- Verify the CUDA version by typing “nvidia-smi” on the Terminal.
- Install pytorch from pytorch.org. Scroll it down to see this section shown below:

For CPU users:

		pip3 install torch torchvision torchaudio 

For GPU users:

		pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

- Copy the command and paste it on the Terminal.
- Install module name ultralytics

To install Ultralytics module, copy and paste this to the terminal:
		
  		pip install ultralytics

You can also visit https://pytorch.org/ to download the version that fits to your CUDA version, python environment, and operating system.

# PREPARING CUSTOM DATASET
- Download the “rock-paper-scissors” dataset from Roboflow [1].
- Select “YOLOv11” download format.
- Select “Download zip to computer”  then click “Continue”.
- Save the dataset to your working folder.

# YAML CONFIGURATION FOR CUSTOM DATASET
- Go to the dataset “Overview”.
- Inspect the classes names under “CLASSES”.
- Go to your working folder and select “data.yaml” file.
- You can see their the directory of your train, valid, and test datasets.
- “nc” is the number of classes and check under “names” if the class names are the same in the overview.

# TRAINING YOLOv11 WITH CUSTOM DATASET
- Training YOLOv11 requires following commands and set of codes to run.
- In the next slide, copy the code shown:

Training the model:

    from ultralytics import YOLO
    
    if __name__ == '__main__':
      
      model = YOLO('yolo11n.pt')  # Start with pre-trained weights
      results = model.train(
          data='replace_with_your_actual_data.yaml_folder_directory',
          epochs=100, #Replace this with our preference
          imgsz=640, # Input image size
          batch=16,  # Adjust batch size 
          patience=50,
          device='0'  # Replace with "0" to use your machine's GPU if pytorch GPU installed. Replace with "cpu" if pytorch CPU installed.
      )
- Copy the directory path of your data.yaml file.
- Paste it on the value of “data” variable on your code.
- Double check the parameter values. You can check other parameters here:
  https://docs.ultralytics.com/modes/train/#train-settings

# RUNNING INFERENCE WITH CUSTOM MODEL
- Using the “YOLODetect.py”, replace model path with your actual directory. 

Find this line on the “YOLODetect.py” python file uploaded.

     model = YOLO('path_to_your_trained_model.pt')
- Run the code. This should use your webcam as the input device.

You can now explore your own projects. Feel free to ask!

References

[1] R. Roboflow, “Rock-paper-scissors Dataset,” Roboflow Universe, Feb. 2025. [Online]. Available: https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw. [Accessed: Feb. 3, 2025].
