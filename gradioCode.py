import os
import gradio as gr
import pandas as pd
import psutil  # Import psutil for hardware information
import GPUtil  # Import GPUtil for GPU information

# Path to the Default.txt file
default_file_path = "Default.txt"

# Ensure the 'results' folder exists
if not os.path.exists('results'):
    os.makedirs('results')

# Function to read default values from Default.txt
def read_default_values():
    if os.path.exists(default_file_path):
        with open(default_file_path, 'r') as f:
            lines = f.readlines()
            return {line.split('=')[0]: line.split('=')[1].strip() for line in lines}
    else:
        return {}

# Function to write default values to Default.txt
def save_default_values(dataset_folder, train_test_split, seed, num_clients, lr, factor, patience, epochs_per_round,
                        initial_lr, step_size, gamma, num_rounds, num_cpus, num_gpus, model_type):
    values = {
        'dataset_folder': dataset_folder,
        'train_test_split': train_test_split,
        'seed': seed,
        'num_clients': num_clients,
        'lr': lr,
        'factor': factor,
        'patience': patience,
        'epochs_per_round': epochs_per_round,
        'initial_lr': initial_lr,
        'step_size': step_size,
        'gamma': gamma,
        'num_rounds': num_rounds,
        'num_cpus': num_cpus,
        'num_gpus': num_gpus,
        'model_type': model_type
    }
    with open(default_file_path, 'w') as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")

    return "Default values saved!"

default_values = read_default_values()

# Function to get hardware information
def get_hardware_info():
    # CPU Info
    cpu_info = f"CPU: {psutil.cpu_count(logical=True)} cores, {psutil.cpu_freq().max:.2f} MHz"

    # Memory Info
    memory_info = psutil.virtual_memory()
    memory_total = f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB"

    # GPU Info
    gpus = GPUtil.getGPUs()
    gpu_info = "No GPU found"
    if gpus:
        gpu_info = [f"GPU: {gpu.name}, Memory: {gpu.memoryTotal} MB" for gpu in gpus]
        gpu_info = ", ".join(gpu_info)

    # Collect all info
    hardware_info = f"{cpu_info}\n{memory_total}\n{gpu_info}"
    return hardware_info

# Function to read resource consumption data from a file and convert to a DataFrame
def read_resource_data(folder_name):
    file_path = os.path.join('results', folder_name, 'resource_consumption.txt')
    if os.path.exists(file_path):
        try:
            # Explicitly define column names
            column_names = ["Round", "CPU Usage (%)", "GPU Usage (%)", "Memory Usage (%)", "Network Sent (MB)", "Network Received (MB)"]
            # Skip the first three lines to get to the data
            df = pd.read_csv(file_path, sep=",", names=column_names, skiprows=3)
            return df
        except pd.errors.ParserError as e:
            print(f"Error parsing file {file_path}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame if parsing fails
    else:
        return pd.DataFrame()

# Function to read aggregated evaluation loss data from a file
def read_aggregated_evaluation_data(folder_name):
    file_path = os.path.join('results', folder_name, 'aggregated_evaluation_loss.txt')
    if os.path.exists(file_path):
        data = {
            "Round": [],
            "Learning Rate (LR)": [],
            "Aggregated Test SSIM": []
        }
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if "Round" in line and "LR" in line:
                        parts = line.strip().split(' - ')
                        round_number = int(parts[1].split(' ')[1])
                        lr = float(parts[2].split(' ')[1])
                        data["Round"].append(round_number)
                        data["Learning Rate (LR)"].append(lr)
                    elif "Aggregated Test SSIM" in line:
                        ssim_value = float(line.strip().split(' ')[-1])
                        data["Aggregated Test SSIM"].append(ssim_value)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# Load default values at startup
default_values = read_default_values()
