from imports import *
from models import SparseAutoencoder
from flower_client import get_parameters
from utils import aggregated_parameters_to_state_dict
import os
from datetime import datetime
import psutil
import GPUtil

class FedCustom(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_lr: float = 0.0005,
        step_size: int = 30,
        gamma: float = 0.9,
    ) -> None:
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.redistributed_parameters = {}
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        self.scheduler = None

        # Create a new subfolder within "results" using the current date and time
        self.results_subfolder = os.path.join("results", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.results_subfolder, exist_ok=True)

        # Initialize the resource consumption log file
        self.resource_consumption_file = os.path.join(self.results_subfolder, "resource_consumption.txt")
        self.initialize_resource_log()

        # Log initial resource consumption as Round 0
        self.log_initial_resource_consumption()

    def initialize_resource_log(self):
        """Initialize the resource consumption log file with column headers."""
        # Get maximum hardware specifications
        cpu_count = psutil.cpu_count(logical=True)
        total_memory = round(psutil.virtual_memory().total / (1024 ** 3), 2)  # in GB
        gpus = GPUtil.getGPUs()
        gpu_name = gpus[0].name if gpus else "N/A"
        total_gpu_memory = round(gpus[0].memoryTotal, 2) if gpus else "N/A"  # in MB

        with open(self.resource_consumption_file, 'w') as file:
            file.write(f"Resource Consumption Log\n")
            file.write(f"CPU (Cores: {cpu_count}), GPU (Model: {gpu_name}, Memory: {total_gpu_memory} MB), Memory (Total: {total_memory} GB), Network (Bytes Sent/Received)\n")
            file.write("Round, CPU Usage (%), GPU Usage (%), Memory Usage (%), Network Sent (MB), Network Received (MB)\n")

    def log_initial_resource_consumption(self):
        """Log the resource consumption at the start of the simulation as Round 0."""
        print("Logging initial resource consumption for Round 0")
        self.log_resource_consumption(0)

    def log_resource_consumption(self, server_round):
        """Log the resource consumption to the file."""
        # Measure CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Measure GPU usage
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0
        
        # Measure Memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # Measure Network usage
        net_io = psutil.net_io_counters()
        net_sent = round(net_io.bytes_sent / (1024 ** 2), 2)  # Convert to MB
        net_received = round(net_io.bytes_recv / (1024 ** 2), 2)  # Convert to MB

        # Log the data into the file
        with open(self.resource_consumption_file, 'a') as file:
            file.write(f"{server_round}, {cpu_usage}, {gpu_usage}, {memory_usage}, {net_sent}, {net_received}\n")


    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters using Autoencoder."""
        net = SparseAutoencoder()
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:
        """Configure the next round of training with redistributed models."""
        num_clients = len(client_manager)
        if num_clients < self.min_fit_clients:
            return []

        sample_size = int(num_clients * self.fraction_fit)
        sample_size = max(sample_size, self.min_fit_clients)
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_fit_clients)

        fit_configurations = [
            (client, FitIns(parameters, {"server_round": server_round}))
            for client in clients
        ]
        return fit_configurations

    def aggregate_parameters(self, parameters_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Aggregate model parameters by averaging them."""
        num_models = len(parameters_list)
        aggregated_parameters = []
        for param_tuple in zip(*parameters_list):
            aggregated_param = np.mean(param_tuple, axis=0)
            aggregated_parameters.append(aggregated_param)
        return aggregated_parameters

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client model updates and prepare the global model for redistribution."""
        if not results:
            return None, {}

        # Log resource consumption
        self.log_resource_consumption(server_round)

        # Collect parameters from the results
        parameters_list = [parameters_to_ndarrays(res.parameters) for client, res in results]
        aggregated_parameters = self.aggregate_parameters(parameters_list)
        aggregated_parameters_fl = ndarrays_to_parameters(aggregated_parameters)

        # Save the latest model's state_dict
        net = SparseAutoencoder()
        state_dict = aggregated_parameters_to_state_dict(aggregated_parameters)
        net.load_state_dict(state_dict)
        torch.save(net.state_dict(), os.path.join(self.results_subfolder, "latest_model.pth"))

        return aggregated_parameters_fl, {}

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation or reconstruction."""
        if self.fraction_evaluate == 0.0:
            return []

        config = {"server_round": server_round, "task": "evaluate"}

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        evaluate_ins = EvaluateIns(parameters, config=config)

        return [(client, evaluate_ins) for client in clients]
    
    def log_all_clients_hardware_resources(self, server_round, client_results):
        """Log the hardware resource consumption for all clients in a single file."""
        # Create or append to the central hardware log file
        hardware_file_path = os.path.join(self.results_subfolder, 'hardware_resources.ncol')

        with open(hardware_file_path, 'a') as file:
            file.write(f"Round {server_round}\n")  # Write the round number

            for client, res in client_results:
                # Log CPU, GPU, memory, and network usage for each client
                cpu_usage = psutil.cpu_percent(interval=1)
                gpus = GPUtil.getGPUs()
                gpu_usage = gpus[0].load * 100 if gpus else 0
                memory_usage = psutil.virtual_memory().percent
                net_io = psutil.net_io_counters()
                net_sent = round(net_io.bytes_sent / (1024 ** 2), 2)  # Convert to MB
                net_received = round(net_io.bytes_recv / (1024 ** 2), 2)  # Convert to MB

                # Log the client's resource usage with their correct client ID
                file.write(f"Client {client.cid}: CPU {cpu_usage}%, GPU {gpu_usage}%, Memory {memory_usage}%, Network Sent: {net_sent}MB, Network Received: {net_received}MB\n")


    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]], failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results, log SSIM, and client hardware resource consumption."""
        if not results:
            return None, {}

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ssim_file_path = os.path.join(self.results_subfolder, 'ssim_scores.ncol')
        evaluation_file_path = os.path.join(self.results_subfolder, 'aggregated_evaluation_loss.txt')

        ssim_scores = []
        total_ssim = 0.0
        total_examples = 0

        # Collect SSIM scores and log resource consumption
        self.log_all_clients_hardware_resources(server_round, results)

        # Collect SSIM scores
        for client, res in results:
            if 'ssim' in res.metrics:
                ssim_scores.append((client.cid, res.metrics['ssim']))
            total_ssim += res.metrics['ssim'] * res.num_examples
            total_examples += res.num_examples

        aggregated_ssim = total_ssim / total_examples if total_examples > 0 else None

        # Sort SSIM scores by client ID and append new data to the SSIM file
        ssim_scores.sort(key=lambda x: int(x[0]))
        with open(ssim_file_path, 'a') as file:
            file.write(f"Time: {current_time} - Round {server_round}\n")
            for cid, ssim_value in ssim_scores:
                file.write(f"{cid} {ssim_value}\n")

        # Update scheduler and append evaluation results
        if self.scheduler is None:
            self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor([1.0]))], lr=self.initial_lr)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

        if aggregated_ssim is not None:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            with open(evaluation_file_path, 'a') as file:
                file.write(f"Time: {current_time} - Round {server_round} - LR {current_lr}\n")
                if aggregated_ssim is not None:
                    file.write(f"Aggregated Test SSIM: {aggregated_ssim:.4f}\n")
                    print(f"Saved aggregated SSIM value for round {server_round} in aggregated_evaluation_loss.txt")

        return aggregated_ssim, {}



    def evaluate(
        self, server_round: int, parameters: fl.common.Parameters
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

def aggregated_parameters_to_state_dict(aggregated_parameters):
    """Convert aggregated parameters to a state dictionary."""
    state_dict = {}
    param_keys = list(SparseAutoencoder().state_dict().keys())
    for key, param in zip(param_keys, aggregated_parameters):
        state_dict[key] = torch.tensor(param)
    return state_dict