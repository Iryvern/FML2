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


