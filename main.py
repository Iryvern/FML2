import os
from datasets import load_datasets  # Import load_datasets function
from strategy import FedCustom  # Import FedCustom strategy
from flower_client import client_fn  # Import client_fn
import flwr as fl  # Import Flower
import warnings
from torchvision import transforms
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gradio as gr
from gradioCode import *

# Define model type variable
model_type = ""

def setup_gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Federated Learning Simulation UI")
        
        # Create a refresh button
        refresh_button = gr.Button("Refresh Folders", variant="secondary", elem_id="refresh-btn", interactive=True)

        # Create tabs for variables, monitoring, and info
        with gr.Tabs():
            with gr.TabItem("Variables"):
                with gr.Row():
                    model_type_input = gr.Dropdown(
                        choices=["Image Anomaly Detection", "Image Classification"],
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

            with gr.TabItem("Hardware Monitoring"):
                gr.Markdown("### Hardware Resource Monitoring")
                
                # List the folders in "results"
                def get_results_folders():
                    return [folder for folder in os.listdir('results') if os.path.isdir(os.path.join('results', folder))]

                folder_list_hardware = gr.Dropdown(
                    choices=get_results_folders(),
                    label="Select Results Folder",
                    interactive=True
                )

                # Display table for selected folder's resource monitoring
                resource_table = gr.DataFrame(headers=["Round", "CPU Usage (%)", "GPU Usage (%)"], visible=False)

                # Add components to display the CPU and GPU resource plots
                cpu_plot = gr.Image(type="filepath")
                gpu_plot = gr.Image(type="filepath")

                # Update the table and plots when a folder is selected
                def update_hardware_table_and_plots(folder_name):
                    # Read the resource data and generate the plots
                    resource_df = read_resource_data(folder_name)
                    plot_paths = plot_hardware_resource_consumption(os.path.join('results', folder_name, 'hardware_resources.ncol'))
                    
                    # Return updated table and plot paths
                    return (
                        gr.update(value=resource_df, visible=True),
                        plot_paths['CPU'],
                        plot_paths['GPU']
                    )

                # Set up the listener to update table and plots on folder selection
                folder_list_hardware.change(
                    fn=update_hardware_table_and_plots, 
                    inputs=folder_list_hardware, 
                    outputs=[resource_table, cpu_plot, gpu_plot]
                )

            with gr.TabItem("Performance Monitoring"):
                gr.Markdown("### Performance Monitoring")

                folder_list_performance = gr.Dropdown(
                    choices=get_results_folders(),
                    label="Select Results Folder",
                    interactive=True
                )

                resource_table = gr.DataFrame(headers=["Round", "CPU Usage (%)", "GPU Usage (%)"], visible=False)
                evaluation_table = gr.DataFrame(headers=["Round", "Learning Rate (LR)", "Aggregated Test SSIM"], visible=False)

                # Update the tables and plot when a folder is selected
                def update_performance(folder_name):
                    resource_df = read_resource_data(folder_name)
                    evaluation_df = read_aggregated_evaluation_data(folder_name)
                    
                    # Get path to the plot image
                    plot_image_path = plot_ssim_scores(os.path.join('results', folder_name, 'ssim_scores.ncol'))
                    
                    return gr.update(value=resource_df, visible=True), gr.update(value=evaluation_df, visible=True), plot_image_path

                # Add the graph component
                plot_output = gr.Image(type="filepath")  # Use "filepath" for the image file

                # Set up the change listener to update tables and plot
                folder_list_performance.change(
                    fn=update_performance,
                    inputs=folder_list_performance,
                    outputs=[resource_table, evaluation_table, plot_output]
                )

            with gr.TabItem("Info"):
                gr.Markdown("### Hardware Information")
                hardware_info = gr.Textbox(value=get_hardware_info(), label="System Hardware Information", lines=5)

        # Refresh button action to update folder dropdowns
        def refresh_folder_lists():
            new_choices = get_results_folders()
            return gr.update(choices=new_choices), gr.update(choices=new_choices)

        # Set refresh button to trigger the folder update in both dropdowns
        refresh_button.click(
            fn=refresh_folder_lists, 
            inputs=[], 
            outputs=[folder_list_hardware, folder_list_performance]
        )

    return demo

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

    # Define transformations based on model type
    if model_type == "Image Anomaly Detection":
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
    else:  # For "Image Classification" model, use a different transformation
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # Load the dataset with user-provided parameters
    trainloaders, testloader = load_datasets(num_clients, dataset_folder, train_transform, test_transform, model_type)

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

if __name__ == "__main__":
    # Launch the Gradio UI
    demo = setup_gradio_ui()
    demo.launch(share=True)
