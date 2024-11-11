from imports import *
from models import SparseAutoencoder, MobileNetV3
import pandas as pd
import os

def aggregated_parameters_to_state_dict(aggregated_parameters, model_type="Image Classification"):
    state_dict = {}

    # Choose model parameters based on model_type
    if model_type == "Image Anomaly Detection":
        param_keys = list(SparseAutoencoder().state_dict().keys())
    elif model_type == "Image Classification":
        param_keys = list(MobileNetV3().state_dict().keys())
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    for key, param in zip(param_keys, aggregated_parameters):
        state_dict[key] = torch.tensor(param)
    return state_dict

def clear_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.truncate(0)

def poison_dataset(dataset_path, config_path='Default.txt'):
    # Read poison_percentage from the config file
    with open(config_path, 'r') as f:
        config = dict(line.strip().split('=') for line in f if '=' in line)
    
    poison_value = float(config.get('poison_percentage', 0))

    if poison_value > 0:
        # Read the dataset
        dataset = pd.read_csv(dataset_path)
        
        # Get the number of rows in the dataset
        row_count = len(dataset)
        
        # Calculate the number of rows to poison
        poison_count = int((poison_value / 100) * row_count)
        
        # Create a copy of the dataset
        new_path = dataset_path.replace('.csv', '_poisoned.csv')
        poisoned_dataset = dataset.copy()
        
        # Increment 'class_id' for the selected percentage of rows
        rows_to_poison = poisoned_dataset.sample(n=poison_count, random_state=42).index
        poisoned_dataset.loc[rows_to_poison, 'class_id'] += 1
        
        # Save the poisoned dataset
        poisoned_dataset.to_csv(new_path, index=False)
        print(poison_count," rows were poisoned")
        return new_path

    print(poison_value," percent of the dataset was infected")
    # If poison_value is 0, return the original dataset path
    return dataset_path
