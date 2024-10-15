from imports import *
from models import SparseAutoencoder,GRUAnomalyDetector

def aggregated_parameters_to_state_dict(aggregated_parameters, model_type="image"):
    state_dict = {}
    if model_type == "image":
        param_keys = list(SparseAutoencoder().state_dict().keys())
    elif model_type in ["gps", "time_series_anomaly_detection"]:
        input_size = 6
        hidden_size = 128
        num_layers = 2
        output_size = 6
        param_keys = list(GRUAnomalyDetector(input_size, hidden_size, num_layers, output_size).state_dict().keys())
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Please use 'image', 'gps', or 'time_series_anomaly_detection'.")

    for key, param in zip(param_keys, aggregated_parameters):
        state_dict[key] = torch.tensor(param)
    return state_dict

def clear_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.truncate(0)

            
