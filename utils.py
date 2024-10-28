from imports import *
from models import SparseAutoencoder, SimpleCNN  # Replace YOLOv11 with SimpleCNN

def aggregated_parameters_to_state_dict(aggregated_parameters, model_type="autoencoder"):
    state_dict = {}

    # Choose model parameters based on model_type
    if model_type == "autoencoder":
        param_keys = list(SparseAutoencoder().state_dict().keys())
    elif model_type == "Image Classification":
        param_keys = list(SimpleCNN().state_dict().keys())
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    for key, param in zip(param_keys, aggregated_parameters):
        state_dict[key] = torch.tensor(param)
    return state_dict

def clear_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.truncate(0)
