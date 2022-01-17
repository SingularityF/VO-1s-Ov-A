# Select a model to use
model_path = "./models/iter2_2.4.h5"
#model_path = "./models/test.h5"

# AWS settings
aws_region = "us-east-1"
credentials_file = "./credentials.csv"

# Default folders
img_output_path = "./saved_img"
img_folder_path = "./unlabeled_data"
dataset_folder_path = "./datasets"
model_folder_path = "./models"
presets_path = "./presets"

# OCR settings, controls screen capture frame rate
#   Determines how many loops in a second, prevent frozen while loop
loop_time = .01
#   Determines how many OCR performed in a second
refresh_time = .2
#   Determines how long to wait then retry if given no/bad input
wait_time = 3

# Hotkey settings
pause_key = 'f9'
capture_key = 'f8'
exit_key = 'f12'