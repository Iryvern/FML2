from imports import *
from models import SparseAutoencoder, MobileNetV3
from flower_client import get_parameters
from utils import aggregated_parameters_to_state_dict
import os
from datetime import datetime
import psutil
import GPUtil
import h5py
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
        model_type: str = "Image Classification",
        num_clusters: int = 3,
    ) -> None:
        with open('Default.txt', 'r') as f:
            config = dict(line.strip().split('=') for line in f if '=' in line)

        dynamic_grouping = float(config.get('dynamic_grouping', 0))
        clustering_frequency = int(config.get('clustering_frequency', 1))  # Fetch the correct frequency value

        self.dynamic_grouping = dynamic_grouping
        self.clustering_frequency = clustering_frequency
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
        self.model_type = model_type
        self.num_clusters = num_clusters  # Fixed number of clusters
        self.cluster_labels = None
        self.cluster_models = {cluster: None for cluster in range(self.num_clusters)}

        # Create a new subfolder within "results" using model type, date, and time
        self.results_subfolder = os.path.join(
            "results", f"{self.model_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        os.makedirs(self.results_subfolder, exist_ok=True)

        # Initialize the resource consumption log file
        self.resource_consumption_file = os.path.join(self.results_subfolder, "resource_consumption.txt")
        self.initialize_resource_log()


    def initialize_resource_log(self):
        """Initialize the resource consumption log file with column headers."""
        cpu_count = psutil.cpu_count(logical=True)
        total_memory = round(psutil.virtual_memory().total / (1024 ** 3), 3)  # in GB
        gpus = GPUtil.getGPUs()
        gpu_name = gpus[0].name if gpus else "N/A"
        total_gpu_memory = round(gpus[0].memoryTotal, 3) if gpus else "N/A"  # in MB

        with open(self.resource_consumption_file, 'w') as file:
            file.write(f"Resource Consumption Log\n")
            file.write(f"CPU (Cores: {cpu_count}), GPU (Model: {gpu_name}, Memory: {total_gpu_memory} MB), Memory (Total: {total_memory} GB), Network (Bytes Sent/Received)\n")
            file.write("Round, Aggregated CPU Usage (%), Aggregated GPU Usage (%), Avg Memory Usage (%), Avg Network Sent (MB), Avg Network Received (MB)\n")

    def log_resource_consumption(self, server_round, client_metrics):
        """Aggregate and log client resource consumption for the round."""
        total_cpu = sum(metric["cpu"] for metric in client_metrics)
        total_gpu = sum(metric["gpu"] for metric in client_metrics)
        avg_memory = round(sum(metric["memory"] for metric in client_metrics) / len(client_metrics), 3)
        avg_net_sent = round(sum(metric["net_sent"] for metric in client_metrics) / len(client_metrics), 3)
        avg_net_received = round(sum(metric["net_received"] for metric in client_metrics) / len(client_metrics), 3)

        with open(self.resource_consumption_file, 'a') as file:
            file.write(f"{server_round}, {round(total_cpu, 3)}, {round(total_gpu, 3)}, {avg_memory}, {avg_net_sent}, {avg_net_received}\n")

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters based on the model type."""
        if self.model_type == "Image Anomaly Detection":
            net = SparseAutoencoder()
        elif self.model_type == "Image Classification":
            net = MobileNetV3()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:
        """Configure the next round of training with optional dynamic grouping."""
        num_clients = len(client_manager)
        if num_clients < self.min_fit_clients:
            return []

        sample_size = int(num_clients * self.fraction_fit)
        sample_size = max(sample_size, self.min_fit_clients)
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=self.min_fit_clients)

        fit_configurations = []
        for client in clients:
            client_id = int(client.cid)
            
            # Apply dynamic grouping logic only if enabled
            if self.dynamic_grouping == 1 and server_round > 1 and hasattr(self, 'cluster_labels') and self.cluster_labels is not None:
                cluster = self.cluster_labels[client_id % len(self.cluster_labels)]
                cluster_parameters = self.cluster_models[cluster]
            else:
                cluster_parameters = parameters

            fit_configurations.append((client, FitIns(cluster_parameters, {"server_round": server_round})))

        return fit_configurations

    def aggregate_parameters(self, parameters_list: List[List[np.ndarray]], server_round: int) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Aggregate model parameters with optional dynamic grouping."""
        num_models = len(parameters_list)
        cluster_labels = None

        if self.dynamic_grouping == 1:
            # Apply clustering only on the first round or at intervals of clustering_frequency
            if server_round == 1 or server_round % self.clustering_frequency == 0:
                # Flatten the parameter arrays to create a feature vector for each model
                flattened_parameters = [np.concatenate([param.flatten() for param in params]) for params in parameters_list]

                # Perform clustering using KMeans based on cosine similarity
                similarity_matrix = cosine_similarity(flattened_parameters)
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init='auto')
                cluster_labels = kmeans.fit_predict(similarity_matrix)
            else:
                # Load cluster labels from the last clustering round if available
                cluster_assignment_file_path = os.path.join(self.results_subfolder, 'cluster_assignments.h5')
                if os.path.exists(cluster_assignment_file_path):
                    with h5py.File(cluster_assignment_file_path, 'r') as f:
                        previous_rounds = [int(r) for r in f.keys() if int(r) < server_round]
                        if previous_rounds:
                            last_round = max(previous_rounds)
                            cluster_labels = f[str(last_round)]['cluster_labels'][:]
                        else:
                            cluster_labels = np.zeros(num_models, dtype=int)
                else:
                    cluster_labels = np.zeros(num_models, dtype=int)

            # Aggregate parameters within each cluster
            aggregated_parameters = []
            for cluster in range(self.num_clusters):
                cluster_parameters = [parameters_list[i] for i in range(num_models) if cluster_labels[i] == cluster]
                if cluster_parameters:
                    cluster_aggregated_parameters = [np.mean(np.array(param_tuple), axis=0) for param_tuple in zip(*cluster_parameters)]
                    aggregated_parameters.append(cluster_aggregated_parameters)

            # Further aggregate the cluster centers to obtain the final parameters
            if aggregated_parameters:
                final_aggregated_parameters = [np.mean(np.array(param_tuple), axis=0) for param_tuple in zip(*aggregated_parameters)]
            else:
                final_aggregated_parameters = [np.zeros_like(param) for param in parameters_list[0]]

            # Update the cluster models for the next round
            self.cluster_models = {cluster: fl.common.ndarrays_to_parameters(params) for cluster, params in enumerate(aggregated_parameters)}
        else:
            # Default global aggregation
            final_aggregated_parameters = [np.mean(param_tuple, axis=0) for param_tuple in zip(*parameters_list)]

        return final_aggregated_parameters, cluster_labels



    def _save_cluster_assignments(self, results, cluster_labels, server_round):
        """Save the cluster assignments for each client in a single file with sorted client IDs."""
        if self.dynamic_grouping != 1 or cluster_labels is None:
            return  # Skip saving if dynamic grouping is not enabled or cluster_labels is None.

        cluster_assignment_file_path = os.path.join(self.results_subfolder, 'cluster_assignments.h5')
        consolidated_log_path = os.path.join(self.results_subfolder, "cluster_assignments.txt")
        client_ids = [client.cid for client, _ in results]

        # Combine client IDs and cluster labels, then sort by client ID
        sorted_assignments = sorted(zip(client_ids, cluster_labels), key=lambda x: int(x[0]))

        # Separate sorted client IDs and cluster labels
        sorted_client_ids, sorted_cluster_labels = zip(*sorted_assignments)

        # Save cluster assignments in the HDF5 file
        with h5py.File(cluster_assignment_file_path, 'a') as f:
            if str(server_round) not in f:
                grp = f.create_group(str(server_round))
                grp.create_dataset("client_ids", data=np.array(sorted_client_ids, dtype='i'))
                grp.create_dataset("cluster_labels", data=np.array(sorted_cluster_labels, dtype='i'))
            else:
                grp = f[str(server_round)]
                grp["client_ids"][:] = np.array(sorted_client_ids, dtype='i')
                grp["cluster_labels"][:] = np.array(sorted_cluster_labels, dtype='i')

        # Append sorted cluster assignments to a single text file
        with open(consolidated_log_path, 'a') as log_file:
            log_file.write(f"\nRound {server_round} Cluster Assignments:\n")
            log_file.write(f"{'Client ID':<15} {'Cluster Label':<15}\n")
            for cid, label in sorted_assignments:
                log_file.write(f"{cid:<15} {label:<15}\n")

        print(f"Cluster assignments for round {server_round} saved to a consolidated file with sorted client IDs.")


    def aggregate_fit(
        self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], 
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results and save models for both dynamic and default grouping."""

        if not results:
            return None, {}

        # Convert FitRes to list of parameter ndarrays and extract local models
        parameters_list = [parameters_to_ndarrays(res.parameters) for client, res in results]
        local_updates = [{"client_id": client.cid, "model": res.parameters} for client, res in results]

        # Detect poisoned clients using CosDefense
        if self.dynamic_grouping == 1:
            # Convert global parameters into the model
            global_parameters_ndarrays = parameters_to_ndarrays(self.initialize_parameters(None))
            if self.model_type == "Image Anomaly Detection":
                global_model = SparseAutoencoder()
            elif self.model_type == "Image Classification":
                global_model = MobileNetV3()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Load the global parameters into the model
            global_model.load_state_dict(aggregated_parameters_to_state_dict(global_parameters_ndarrays, self.model_type))
            self.detect_potential_poisoned_client(server_round, global_model, local_updates)

        # Dynamic grouping logic if enabled
        if self.dynamic_grouping == 1:
            aggregated_parameters, cluster_labels = self.aggregate_parameters(parameters_list, server_round)
            self.cluster_labels = cluster_labels  # Store cluster_labels for use in other functions

            # Save the latest model for each cluster
            for cluster, params in self.cluster_models.items():
                if self.model_type == "Image Anomaly Detection":
                    net = SparseAutoencoder()
                elif self.model_type == "Image Classification":
                    net = MobileNetV3()

                state_dict = aggregated_parameters_to_state_dict(parameters_to_ndarrays(params), self.model_type)
                net.load_state_dict(state_dict)
                torch.save(net.state_dict(), os.path.join(self.results_subfolder, f"latest_model_cluster_{cluster}.pth"))
        else:
            # Default global aggregation logic
            aggregated_parameters = [np.mean(param_tuple, axis=0) for param_tuple in zip(*parameters_list)]

            # Save the aggregated model (global model) only for default grouping
            if self.model_type == "Image Anomaly Detection":
                net = SparseAutoencoder()
            elif self.model_type == "Image Classification":
                net = MobileNetV3()

            state_dict = aggregated_parameters_to_state_dict(aggregated_parameters, self.model_type)
            net.load_state_dict(state_dict)
            torch.save(net.state_dict(), os.path.join(self.results_subfolder, "latest_model.pth"))

        aggregated_parameters_fl = fl.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters_fl, {}



    def configure_evaluate(
            self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.ClientManager
        ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """Configure evaluation with optional dynamic grouping."""
        
        if self.fraction_evaluate == 0.0:
            return []

        config = {"server_round": server_round, "task": "evaluate"}

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        if self.dynamic_grouping == 1 and server_round > 1 and self.cluster_labels is not None:
            # Assign cluster-specific parameters to clients based on clustering
            evaluate_configurations = []
            for client in clients:
                client_id = client.cid
                cluster = self.cluster_labels[int(client_id) % len(self.cluster_labels)]
                cluster_parameters = self.cluster_models[cluster]
                evaluate_ins = fl.common.EvaluateIns(cluster_parameters, config=config)
                evaluate_configurations.append((client, evaluate_ins))
            return evaluate_configurations

        # Default evaluation configuration
        evaluate_ins = fl.common.EvaluateIns(parameters, config=config)
        return [(client, evaluate_ins) for client in clients]

    
    def log_all_clients_hardware_resources(self, server_round, client_results):
        """Log each client's hardware usage in hardware_resources.ncol and aggregate CPU/GPU for resource_consumption.txt."""
        hardware_file_path = os.path.join(self.results_subfolder, 'hardware_resources.ncol')
        client_metrics = []

        with open(hardware_file_path, 'a') as file:
            file.write(f"Round {server_round}\n")
            for client, res in client_results:
                cpu_usage = round(psutil.cpu_percent(interval=1), 3)
                gpus = GPUtil.getGPUs()
                gpu_usage = round(gpus[0].load * 100, 3) if gpus else 0
                memory_usage = round(psutil.virtual_memory().percent, 3)
                net_io = psutil.net_io_counters()
                net_sent = round(net_io.bytes_sent / (1024 ** 2), 3)
                net_received = round(net_io.bytes_recv / (1024 ** 2), 3)

                client_metrics.append({
                    "cpu": cpu_usage,
                    "gpu": gpu_usage,
                    "memory": memory_usage,
                    "net_sent": net_sent,
                    "net_received": net_received
                })

                file.write(f"Client {client.cid}: CPU {cpu_usage}%, GPU {gpu_usage}%, Memory {memory_usage}%, Network Sent: {net_sent}MB, Network Received: {net_received}MB\n")

        # After logging each client's data, log the aggregated metrics
        self.log_resource_consumption(server_round, client_metrics)

    def _compute_group_metrics(self, results):
        """Compute average metric for each group in dynamic grouping."""
        group_metrics = [0.0] * self.num_clusters
        group_counts = [0] * self.num_clusters

        for client, res in results:
            cluster_idx = self.cluster_labels[int(client.cid) % len(self.cluster_labels)]
            metric_value = res.metrics['accuracy'] if self.model_type == "Image Classification" else res.metrics['ssim']
            group_metrics[cluster_idx] += metric_value * res.num_examples
            group_counts[cluster_idx] += res.num_examples

        for idx in range(self.num_clusters):
            if group_counts[idx] > 0:
                group_metrics[idx] /= group_counts[idx]

        return group_metrics

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results and save metrics, supporting dynamic grouping."""

        if not results:
            return None, {}

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metric_file_path = os.path.join(self.results_subfolder, 'accuracy_scores.ncol')
        
        # Determine output file based on dynamic grouping
        evaluation_file_name = 'evaluation_loss.txt' if self.dynamic_grouping == 1 else 'aggregated_evaluation_loss.txt'
        evaluation_file_path = os.path.join(self.results_subfolder, evaluation_file_name)

        total_metric = 0.0
        total_examples = 0
        metric_scores = []
        self.log_all_clients_hardware_resources(server_round, results)

        # Aggregate client metrics
        for client, res in results:
            if self.model_type == "Image Classification" and 'accuracy' in res.metrics:
                metric_scores.append((client.cid, res.metrics['accuracy']))
                total_metric += res.metrics['accuracy'] * res.num_examples
            elif self.model_type == "Image Anomaly Detection" and 'ssim' in res.metrics:
                metric_scores.append((client.cid, res.metrics['ssim']))
                total_metric += res.metrics['ssim'] * res.num_examples
            total_examples += res.num_examples

        aggregated_metric = total_metric / total_examples if total_examples > 0 else None

        # Save client scores
        metric_scores.sort(key=lambda x: int(x[0]))
        with open(metric_file_path, 'a') as file:
            file.write(f"Time: {current_time} - Round {server_round}\n")
            for cid, metric_value in metric_scores:
                file.write(f"{cid} {metric_value}\n")

        # Save grouped or aggregated metrics
        with open(evaluation_file_path, 'a') as file:
            file.write(f"Time: {current_time} - Round {server_round}\n")
            if self.dynamic_grouping == 1 and self.cluster_labels is not None:
                group_metrics = self._compute_group_metrics(results)
                for group_idx, metric in enumerate(group_metrics, start=1):
                    file.write(f"Group-{group_idx}: {metric:.4f}\n")
            else:
                file.write(f"Aggregated Metric: {aggregated_metric:.4f}\n")

        # Save cluster assignments only if dynamic grouping is enabled
        if self.dynamic_grouping == 1:
            self._save_cluster_assignments(results, self.cluster_labels, server_round)

        return aggregated_metric, {}


    def evaluate(
        self, server_round: int, parameters: fl.common.Parameters
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for fitting."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def detect_potential_poisoned_client(self, server_round: int, global_model, local_updates):
        """Detect the client with potentially poisoned data and save the results using CosDefense."""

        # Extract the last layer of the global model
        global_last_layer = next(reversed(global_model.state_dict().values()))

        # Extract local updates
        client_scores = {}
        for update in local_updates:
            client_id = update["client_id"]
            local_parameters = parameters_to_ndarrays(update["model"])
            local_model_state = aggregated_parameters_to_state_dict(local_parameters, self.model_type)

            # Compute cosine similarity
            local_last_layer = next(reversed(local_model_state.values()))
            similarity = cosine_similarity(global_last_layer.reshape(1, -1), local_last_layer.reshape(1, -1))[0][0]
            client_scores[client_id] = similarity

        # Identify the client with the lowest similarity score
        potential_poisoned_client = min(client_scores, key=client_scores.get)

        # Save detection results
        poisoned_log_path = os.path.join(self.results_subfolder, "poisoned_client_detection.txt")
        with open(poisoned_log_path, 'a') as log_file:
            log_file.write(f"Round {server_round} - Potential Poisoned Client Detection\n")
            log_file.write(f"Potential Poisoned Client: Client-{potential_poisoned_client}\n")
            log_file.write(f"Similarity Scores: {client_scores}\n")

        print(f"Detection results saved for round {server_round} in {poisoned_log_path}.")


def aggregated_parameters_to_state_dict(aggregated_parameters, model_type):
    """Convert aggregated parameters to a state dictionary based on model type."""
    if model_type == "Image Anomaly Detection":
        param_keys = list(SparseAutoencoder().state_dict().keys())
    elif model_type == "Image Classification":
        param_keys = list(MobileNetV3().state_dict().keys())
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    state_dict = {key: torch.tensor(param) for key, param in zip(param_keys, aggregated_parameters)}
    return state_dict
