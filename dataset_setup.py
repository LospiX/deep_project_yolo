import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Specify dataset information
dataset_repo = 'kili-technology/plastic_in_river'  # Replace with your dataset repo
dataset_filename = "dataset_file.zip"            # Replace with the filename in the repo
local_dataset_path = "./data"                    # Local path to store the dataset

# Download dataset from Hugging Face
def download_dataset(repo, filename, local_path):
    os.makedirs(local_path, exist_ok=True)
    downloaded_file = hf_hub_download(repo_id=repo, filename=filename, cache_dir=local_path)
    return downloaded_file

# Unzip dataset if necessary
def unzip_file(filepath, extract_to):
    import zipfile
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Define YOLO training configuration
def configure_yolo():
    model = YOLO("yolov8n.pt")  # Choose YOLO version (e.g., yolov8n.pt for nano, yolov8s.pt for small)
    return model

# Main function to execute training
def train_yolo(model, data_path, epochs=50, batch_size=16, img_size=640):
    model.train(
        data=data_path,     # Path to the dataset YAML file
        epochs=epochs,      # Number of training epochs
        batch=batch_size,   # Batch size
        imgsz=img_size      # Image size
    )

# Evaluate the model
def evaluate_model(model, data_path):
    results = model.val(data=data_path)
    return results

# Load dataset
dataset_file = download_dataset(dataset_repo, dataset_filename, local_dataset_path)
unzip_file(dataset_file, local_dataset_path)

# Prepare YOLO model
# yolo_model = configure_yolo()

# Train YOLO model
# train_yolo(yolo_model, data_path=f"{local_dataset_path}/data.yaml", epochs=50)

# Evaluate YOLO model
# evaluation_results = evaluate_model(yolo_model, data_path=f"{local_dataset_path}/data.yaml")

# Display results
# print("Evaluation Results:", evaluation_results)