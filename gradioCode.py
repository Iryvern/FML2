import os
import gradio as gr
import pandas as pd
import psutil
import GPUtil
import matplotlib.pyplot as plt
import tempfile

default_file_path = "Default.txt"

if not os.path.exists('results'):
    os.makedirs('results')

def read_default_values():
    if os.path.exists(default_file_path):
        with open(default_file_path, 'r') as f:
            lines = f.readlines()
            return {line.split('=')[0]: line.split('=')[1].strip() for line in lines}
    else:
        return {}

metrics_to_plot = ['CPU', 'GPU']

def plot_hardware_resource_consumption(file_path: str):
    import matplotlib.pyplot as plt
    import tempfile

    client_metrics = {}

    with open(file_path, 'r') as file:
        current_round = None
        for line in file:
            line = line.strip()
            if line.startswith('Round'):
                current_round = int(line.split()[1])
            elif line.startswith('Client'):
                client_info, metrics_info = line.split(': ', 1)
                client_id = client_info.strip().split()[1]
                metrics = metrics_info.strip().split(', ')
                metrics_dict = {}
                for metric in metrics:
                    if ':' in metric:
                        key, value = metric.split(':')
                        key = key.strip()
                        value = value.strip()
                    else:
                        key_value = metric.strip().split()
                        if len(key_value) >= 2:
                            key = key_value[0]
                            value = key_value[1]
                        else:
                            continue
                    metrics_dict[key] = value

                if client_id not in client_metrics:
                    client_metrics[client_id] = {
                        'rounds': [],
                        'CPU': [],
                        'GPU': [],
                    }

                client_metrics[client_id]['rounds'].append(current_round)
                cpu_usage = float(metrics_dict.get('CPU', '0%').replace('%', ''))
                gpu_usage = float(metrics_dict.get('GPU', '0%').replace('%', ''))
                client_metrics[client_id]['CPU'].append(cpu_usage)
                client_metrics[client_id]['GPU'].append(gpu_usage)

    plot_paths = {}

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        for client_id in client_metrics:
            rounds = client_metrics[client_id]['rounds']
            values = client_metrics[client_id][metric]
            plt.plot(rounds, values, label=f'Client {client_id}')

        plt.xlabel('Round')
        plt.ylabel(f'{metric} Usage (%)')
        plt.title(f'{metric} Usage per Client per Round')
        plt.legend()
        plt.grid(True)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(temp_file.name)
        plt.close()

        plot_paths[metric] = temp_file.name

    return plot_paths

def plot_metric_scores(folder_name):
    """Plot either SSIM or Accuracy scores based on model type in the folder name."""
    if "Image Classification" in folder_name:
        metric_label = "Accuracy"
        y_label = "Accuracy Score"
        file_suffix = "accuracy_scores.ncol"
    else:
        metric_label = "SSIM"
        y_label = "SSIM Score"
        file_suffix = "ssim_scores.ncol"
        
    file_path = os.path.join('results', folder_name, file_suffix)
    round_numbers = []
    client_metrics = {}

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Time'):
                parts = line.split(' - ')
                round_numbers.append(int(parts[1].split()[1]))
            else:
                client_id, score = line.split()
                if client_id not in client_metrics:
                    client_metrics[client_id] = []
                client_metrics[client_id].append(float(score))

    plt.figure(figsize=(12, 8))
    for client_id, scores in client_metrics.items():
        plt.plot(round_numbers, scores, label=f'Client {client_id}')

    plt.xlabel('Round')
    plt.ylabel(y_label)
    plt.title(f'{y_label} per Client per Round')
    plt.legend()
    plt.grid(True)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name)
    plt.close()

    return temp_file.name

def save_default_values(dataset_folder, train_test_split, seed, num_clients, lr, factor, patience, epochs_per_round,
                        initial_lr, step_size, gamma, num_rounds, num_cpus, num_gpus, model_type,poison_percentage, dynamic_grouping):
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
        'model_type': model_type,
        'poison_percentage': poison_percentage,
        'dynamic_grouping': dynamic_grouping,
    }
    with open(default_file_path, 'w') as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")

    return "Default values saved!"


default_values = read_default_values()

def get_hardware_info():
    cpu_info = f"CPU: {psutil.cpu_count(logical=True)} cores, {psutil.cpu_freq().max:.2f} MHz"
    memory_info = psutil.virtual_memory()
    memory_total = f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB"
    gpus = GPUtil.getGPUs()
    gpu_info = "No GPU found"
    if gpus:
        gpu_info = [f"GPU: {gpu.name}, Memory: {gpu.memoryTotal} MB" for gpu in gpus]
        gpu_info = ", ".join(gpu_info)

    hardware_info = f"{cpu_info}\n{memory_total}\n{gpu_info}"
    return hardware_info

def read_resource_data(folder_name):
    file_path = os.path.join('results', folder_name, 'resource_consumption.txt')
    if os.path.exists(file_path):
        try:
            column_names = ["Round", "CPU Usage (%)", "GPU Usage (%)", "Memory Usage (%)", "Network Sent (MB)", "Network Received (MB)"]
            df = pd.read_csv(file_path, sep=",", names=column_names, skiprows=3)
            return df
        except pd.errors.ParserError as e:
            print(f"Error parsing file {file_path}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def read_aggregated_evaluation_data(folder_name):
    file_path = os.path.join('results', folder_name, 'aggregated_evaluation_loss.txt')
    data = {"Round": [], "Learning Rate (LR)": [], "Aggregated Test Metric": []}
    if os.path.exists(file_path):
        metric_label = "Aggregated Test Accuracy" if "Image Classification" in folder_name else "Aggregated Test SSIM"
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
                    elif metric_label in line:
                        metric_value = float(line.strip().split(' ')[-1])
                        data["Aggregated Test Metric"].append(metric_value)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

default_values = read_default_values()