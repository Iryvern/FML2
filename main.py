import gradio as gr
import os
from datasets import load_datasets  # Import load_datasets function
from strategy import FedCustom  # Import FedCustom strategy
from flower_client import client_fn  # Import client_fn
import flwr as fl  # Import Flower
import signal
import sys
import warnings
from torchvision import transforms
warnings.filterwarnings("ignore", category=DeprecationWarning)

if not os.path.exists('results'):
    os.makedirs('results')

# Function to start the training process
def start_training(dataset_folder, train_test_split, seed, num_clients, 
                   lr, factor, patience, epochs_per_round,
                   initial_lr, step_size, gamma, num_rounds, num_cpus, num_gpus):

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
            client_fn=lambda cid: client_fn(cid, trainloaders),  # Adjust to use lambda to pass trainloaders
            num_clients=num_clients, 
            config=fl.server.ServerConfig(num_rounds=num_rounds), 
            strategy=strategy, 
            client_resources={"num_cpus": num_cpus, "num_gpus": num_gpus}
        )
    except Exception as e:
        return f"Error: {e}"

    return "Training started with the provided parameters!"

# Gradio UI setup for starting and stopping the simulation
def setup_gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Federated Learning Simulation UI")

        # Start and Stop buttons at the top
        with gr.Row():
            start_button = gr.Button("Start Simulation")

        output_text = gr.Textbox(label="Output")
        
        # Input fields in 3 columns
        with gr.Row():
            with gr.Column():
                # Use text input for dataset folder path, default to "Dataset"
                dataset_folder_input = gr.Textbox(label="Dataset Folder Path (default: './Dataset')", value=os.path.abspath("Dataset"))
                train_test_split_input = gr.Textbox(label="Train-Test Split (default: 0.8)", value="0.8")
                seed_input = gr.Textbox(label="Seed (default: 42)", value="42")
                num_clients_input = gr.Textbox(label="Number of Clients (default: 10)", value="10")

            with gr.Column():
                lr_input = gr.Textbox(label="Learning Rate (Client) (default: 0.001)", value="0.001")
                factor_input = gr.Textbox(label="Factor (Client) (default: 0.5)", value="0.5")
                patience_input = gr.Textbox(label="Patience (Client) (default: 3)", value="3")
                epochs_input = gr.Textbox(label="Epochs per Round (default: 5)", value="5")

            with gr.Column():
                initial_lr_input = gr.Textbox(label="Initial Learning Rate (FedCustom) (default: 0.001)", value="0.001")
                step_size_input = gr.Textbox(label="Step Size (FedCustom) (default: 30)", value="30")
                gamma_input = gr.Textbox(label="Gamma (FedCustom) (default: 0.9)", value="0.9")
                num_rounds_input = gr.Textbox(label="Number of Rounds (default: 100)", value="100")
                num_cpus_input = gr.Textbox(label="Number of CPUs (default: 1)", value="1")
                num_gpus_input = gr.Textbox(label="Number of GPUs (default: 0.1)", value="0.1")

        # Start button action
        start_button.click(
            start_training, 
            inputs=[
                dataset_folder_input, train_test_split_input, seed_input, num_clients_input, 
                lr_input, factor_input, patience_input, epochs_input,
                initial_lr_input, step_size_input, gamma_input, num_rounds_input,
                num_cpus_input, num_gpus_input
            ], 
            outputs=output_text
        )
    return demo

if __name__ == "__main__":
    # Launch the Gradio UI
    demo = setup_gradio_ui()
    demo.launch(share=True)
