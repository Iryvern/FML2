from imports import *
from models import SparseAutoencoder

def aggregated_parameters_to_state_dict(aggregated_parameters):
    state_dict = {}
    param_keys = list(SparseAutoencoder().state_dict().keys())
    for key, param in zip(param_keys, aggregated_parameters):
        state_dict[key] = torch.tensor(param)
    return state_dict

def clear_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.truncate(0)