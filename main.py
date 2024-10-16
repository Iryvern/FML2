import gradio as gr
import os
import psutil  # Import psutil for hardware information
import GPUtil  # Import GPUtil for GPU information
from datasets import load_datasets  # Import load_datasets function
from strategy import FedCustom  # Import FedCustom strategy
from flower_client import client_fn  # Import client_fn
import flwr as fl  # Import Flower
import pandas as pd  # Import pandas to read the text file and display as a table
import warnings
from torchvision import transforms
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# Load default values at startup
default_values = read_default_values()

# Function to start the training process
def start_training(dataset_folder, train_test_split, seed, num_clients, 
                   lr, factor, patience, epochs_per_round,
                   initial_lr, step_size, gamma, num_rounds, num_cpus, num_gpus, model_type):

    # Cast the inputs to their appropriate types
    train_test_split = float(train_test_split)
    seed = int(seed)
    num_clients = int(num_clients)
    lr = float(lr)
    factor = float(factor)
    patience = int(patience)
    epochs_per_round = int(epochs_per_round)
    initial_lr = float(initial_lr)
    step_size = int(step_size)
    gamma = float(gamma)
    num_rounds = int(num_rounds)
    num_cpus = int(num_cpus)
    num_gpus = float(num_gpus)

    # Define image transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the dataset with user-provided parameters
    trainloaders, testloader = load_datasets(num_clients, dataset_folder, train_transform, test_transform)

    # Set up the strategy with FedCustom
    strategy = FedCustom(
        initial_lr=initial_lr, 
        step_size=step_size, 
        gamma=gamma
    )

    # Start the Flower simulation with the adjusted client_fn logic
    try:
        fl.simulation.start_simulation(
            client_fn=lambda cid: client_fn(cid, trainloaders),
            num_clients=num_clients, 
            config=fl.server.ServerConfig(num_rounds=num_rounds), 
            strategy=strategy, 
            client_resources={"num_cpus": num_cpus, "num_gpus": num_gpus},
            ray_init_args={"include_dashboard": True}
        )
    except Exception as e:
        return f"Error: {e}"

    return "Training started with the provided parameters!"

# Gradio UI setup with tabs for "Variables", "Monitoring", and "Info"
def setup_gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Federated Learning Simulation UI")
        
        # Create tabs for variables, monitoring, and info
        with gr.Tabs():
            with gr.TabItem("Variables"):
                with gr.Row():
                    model_type_input = gr.Dropdown(
                        choices=["Image Anomaly Detection", "Time Series Classification"],
                        label="Model Type", 
                        value=default_values.get('model_type', "Image Anomaly Detection")
                    )
                    start_button = gr.Button("Start Simulation")
                    save_button = gr.Button("Save changes")
                
                output_text = gr.Textbox(label="Output")

                # Input fields in 3 columns with default values from Default.txt
                with gr.Row():
                    with gr.Column():
                        dataset_folder_input = gr.Textbox(
                            label="Dataset Folder Path (default: './Dataset')",
                            value=default_values.get('dataset_folder', os.path.abspath("Dataset"))
                        )
                        train_test_split_input = gr.Textbox(
                            label="Train-Test Split (default: 0.8)",
                            value=default_values.get('train_test_split', "0.8")
                        )
                        seed_input = gr.Textbox(
                            label="Seed (default: 42)", 
                            value=default_values.get('seed', "42")
                        )
                        num_clients_input = gr.Textbox(
                            label="Number of Clients (default: 10)", 
                            value=default_values.get('num_clients', "10")
                        )

                    with gr.Column():
                        lr_input = gr.Textbox(
                            label="Learning Rate (Client) (default: 0.001)", 
                            value=default_values.get('lr', "0.001")
                        )
                        factor_input = gr.Textbox(
                            label="Factor (Client) (default: 0.5)", 
                            value=default_values.get('factor', "0.5")
                        )
                        patience_input = gr.Textbox(
                            label="Patience (Client) (default: 3)", 
                            value=default_values.get('patience', "3")
                        )
                        epochs_input = gr.Textbox(
                            label="Epochs per Round (default: 5)", 
                            value=default_values.get('epochs_per_round', "5")
                        )

                    with gr.Column():
                        initial_lr_input = gr.Textbox(
                            label="Initial Learning Rate (FedCustom) (default: 0.001)", 
                            value=default_values.get('initial_lr', "0.001")
                        )
                        step_size_input = gr.Textbox(
                            label="Step Size (FedCustom) (default: 30)", 
                            value=default_values.get('step_size', "30")
                        )
                        gamma_input = gr.Textbox(
                            label="Gamma (FedCustom) (default: 0.9)", 
                            value=default_values.get('gamma', "0.9")
                        )
                        num_rounds_input = gr.Textbox(
                            label="Number of Rounds (default: 100)", 
                            value=default_values.get('num_rounds', "100")
                        )
                        num_cpus_input = gr.Textbox(
                            label="Number of CPUs (default: 1)", 
                            value=default_values.get('num_cpus', "1")
                        )
                        num_gpus_input = gr.Textbox(
                            label="Number of GPUs (default: 0.1)", 
                            value=default_values.get('num_gpus', "0.1")
                        )

                # Save button action to save current input values to Default.txt
                save_button.click(
                    save_default_values, 
                    inputs=[
                        dataset_folder_input, train_test_split_input, seed_input, num_clients_input,
                        lr_input, factor_input, patience_input, epochs_input,
                        initial_lr_input, step_size_input, gamma_input, num_rounds_input,
                        num_cpus_input, num_gpus_input, model_type_input
                    ], 
                    outputs=output_text
                )

                # Start button action
                start_button.click(
                    start_training, 
                    inputs=[
                        dataset_folder_input, train_test_split_input, seed_input, num_clients_input, 
                        lr_input, factor_input, patience_input, epochs_input,
                        initial_lr_input, step_size_input, gamma_input, num_rounds_input,
                        num_cpus_input, num_gpus_input, model_type_input
                    ], 
                    outputs=output_text
                )

            with gr.TabItem("Monitoring"):
                gr.Markdown("### Resource Consumption Monitoring")
                
                # List the folders in "results"
                def get_results_folders():
                    return [folder for folder in os.listdir('results') if os.path.isdir(os.path.join('results', folder))]

                folder_list = gr.Dropdown(
                    choices=get_results_folders(),
                    label="Select Results Folder",
                    interactive=True
                )

                # Display tables for selected folder
                resource_table = gr.DataFrame(headers=["Round", "CPU Usage (%)", "GPU Usage (%)", "Memory Usage (%)", "Network Sent (MB)", "Network Received (MB)"], visible=False)
                evaluation_table = gr.DataFrame(headers=["Round", "Learning Rate (LR)", "Aggregated Test SSIM"], visible=False)

                # Update the tables when a folder is selected
                def update_tables(folder_name):
                    resource_df = read_resource_data(folder_name)
                    evaluation_df = read_aggregated_evaluation_data(folder_name)
                    return gr.update(value=resource_df, visible=True), gr.update(value=evaluation_df, visible=True)

                folder_list.change(
                    fn=update_tables, 
                    inputs=folder_list, 
                    outputs=[resource_table, evaluation_table]
                )

            with gr.TabItem("Info"):
                gr.Markdown("### Hardware Information")
                hardware_info = gr.Textbox(value=get_hardware_info(), label="System Hardware Information", lines=5)

    return demo

if __name__ == "__main__":
    # Launch the Gradio UI
    demo = setup_gradio_ui()
    demo.launch(share=True)