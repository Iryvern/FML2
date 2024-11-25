import os
from datasets import load_datasets
from strategy import FedCustom
from flower_client import client_fn
import flwr as fl
import warnings
from torchvision import transforms
import gradio as gr
from gradioCode import *
warnings.filterwarnings("ignore", category=DeprecationWarning)

model_type = ""

def setup_gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Federated Learning Simulation UI")

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
                        data_poisoning_percentage_input = gr.Textbox(
                            label="Data Poisoning Percentage (default: 0%)",
                            value=default_values.get('poison_percentage', "0")
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
                        dynamic_grouping_enabled_input = gr.Textbox(
                            label="Dynamic Grouping (Off:0, On:1)",
                            value=default_values.get('dynamic_grouping', "0")
                        )
                        clustering_frequency_input = gr.Textbox(
                            label="Clustering Frequency (Default: 5)",
                            value=default_values.get('clustering_frequency', "5")
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

                save_button.click(
                    save_default_values, 
                    inputs=[
                        dataset_folder_input, train_test_split_input, seed_input, num_clients_input,
                        lr_input, factor_input, patience_input, epochs_input,
                        initial_lr_input, step_size_input, gamma_input, num_rounds_input,
                        num_cpus_input, num_gpus_input, model_type_input,
                        data_poisoning_percentage_input, dynamic_grouping_enabled_input,
                        clustering_frequency_input
                    ], 
                    outputs=output_text
                )

                start_button.click(
                    start_training, 
                    inputs=[
                        dataset_folder_input, train_test_split_input, seed_input, num_clients_input, 
                        lr_input, factor_input, patience_input, epochs_input,
                        initial_lr_input, step_size_input, gamma_input, num_rounds_input,
                        num_cpus_input, num_gpus_input, model_type_input,
                        data_poisoning_percentage_input, dynamic_grouping_enabled_input,
                        clustering_frequency_input
                    ], 
                    outputs=output_text
                )

            with gr.TabItem("Hardware Monitoring"):
                gr.Markdown("### Hardware Resource Monitoring")

                def get_results_folders():
                    return [folder for folder in os.listdir('results') if os.path.isdir(os.path.join('results', folder))]

                folder_list_hardware = gr.Dropdown(
                    choices=get_results_folders(),
                    label="Select Results Folder",
                    interactive=True
                )
                
                refresh_button_hardware = gr.Button("Refresh Folders", variant="secondary", interactive=True)

                resource_table = gr.DataFrame(headers=["Round", "CPU Usage (%)", "GPU Usage (%)"], visible=False)
                cpu_plot = gr.Image(type="filepath")
                gpu_plot = gr.Image(type="filepath")

                def update_hardware_table_and_plots(folder_name):
                    resource_df = read_resource_data(folder_name)
                    plot_paths = plot_hardware_resource_consumption(os.path.join('results', folder_name, 'hardware_resources.ncol'))
                    
                    return (
                        gr.update(value=resource_df, visible=True),
                        plot_paths['CPU'],
                        plot_paths['GPU']
                    )

                folder_list_hardware.change(
                    fn=update_hardware_table_and_plots, 
                    inputs=folder_list_hardware, 
                    outputs=[resource_table, cpu_plot, gpu_plot]
                )
                
                # Connect refresh_button in "Hardware Monitoring" tab
                refresh_button_hardware.click(
                    fn=lambda: gr.update(choices=get_results_folders()),
                    inputs=[],
                    outputs=folder_list_hardware
                )

            with gr.TabItem("Performance Monitoring"):
                gr.Markdown("### Performance Monitoring")

                folder_list_performance = gr.Dropdown(
                    choices=get_results_folders(),
                    label="Select Results Folder",
                    interactive=True
                )
                
                refresh_button_performance = gr.Button("Refresh Folders", variant="secondary", interactive=True)

                evaluation_table = gr.DataFrame(headers=["Round", "Learning Rate (LR)", "Metric"], visible=False)

                def update_performance(folder_name):
                    evaluation_df = read_aggregated_evaluation_data(folder_name)
                    plot_image_path = plot_metric_scores(folder_name)
                    
                    return gr.update(value=evaluation_df, visible=True), plot_image_path

                plot_output = gr.Image(type="filepath")

                folder_list_performance.change(
                    fn=update_performance,
                    inputs=folder_list_performance,
                    outputs=[evaluation_table, plot_output]
                )

                # Connect refresh_button in "Performance Monitoring" tab
                refresh_button_performance.click(
                    fn=lambda: gr.update(choices=get_results_folders()),
                    inputs=[],
                    outputs=folder_list_performance
                )

            with gr.TabItem("Info"):
                gr.Markdown("### Hardware Information")
                hardware_info = gr.Textbox(value=get_hardware_info(), label="System Hardware Information", lines=5)

    return demo


def start_training(dataset_folder, train_test_split, seed, num_clients, 
                   lr, factor, patience, epochs_per_round,
                   initial_lr, step_size, gamma, num_rounds, num_cpus, num_gpus, model_type,data_poisoning_percentage, dynamic_grouping_enabled, clustering_frequency):

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
    elif model_type == "Image Classification":
        train_transform = transforms.Compose([
            transforms.Resize((480, 300)),  # Resize to 480x300
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels (RGB)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((480, 300)),  # Resize to 480x300
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels (RGB)
        ])
    else:
        print("Unrecognized model type for transformation")

    data_split = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
    trainloaders, testloader = load_datasets(num_clients, dataset_folder, train_transform, test_transform, model_type, data_split)

    strategy = FedCustom(
        initial_lr=initial_lr, 
        step_size=step_size, 
        gamma=gamma,
        model_type=model_type
    )

    try:
        fl.simulation.start_simulation(
            client_fn=lambda cid: client_fn(cid, trainloaders, model_type),
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
    demo = setup_gradio_ui()
    demo.launch(share=True)