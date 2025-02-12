from imports import *
from models import SparseAutoencoder, MobileNetV3
from flower_client import get_parameters
from utils import aggregated_parameters_to_state_dict
import re
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
        num_clusters: int = 4,
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
            file.write(f"CPU (Cores: {cpu_count}), GPU (Model: {gpu_name}, Memory: {total_gpu_memory} MB), Memory (Total: {total_memory} GB)\n")
            file.write("Round, Aggregated CPU Usage (%), Aggregated GPU Usage (%)\n")

    def log_resource_consumption(self, server_round, client_metrics):
        """Aggregate and log client resource consumption for the round."""
        total_cpu = sum(metric["cpu"] for metric in client_metrics)
        total_gpu = sum(metric["gpu"] for metric in client_metrics)

        with open(self.resource_consumption_file, 'a') as file:
            file.write(f"{server_round}, {round(total_cpu, 3)}, {round(total_gpu, 3)}\n")

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

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:
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
            if self.dynamic_grouping == 1 and server_round > 1:
                # Ensure client_cluster_mapping is initialized and client_id exists in the mapping
                if hasattr(self, 'client_cluster_mapping') and client_id in self.client_cluster_mapping:
                    cluster = self.client_cluster_mapping[client_id]
                    cluster_parameters = self.cluster_models[cluster]
                else:
                    # If client_id is not in the mapping, use default parameters
                    cluster_parameters = parameters
            else:
                cluster_parameters = parameters

            fit_configurations.append((client, FitIns(cluster_parameters, {"server_round": server_round})))

        return fit_configurations

    def aggregate_parameters(self, parameters_list: List[List[np.ndarray]], server_round: int, client_metrics) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Aggregate model parameters with accuracy-based clustering. Ensure each cluster has at least one client in round 1 and maintain balance in later rounds."""
        num_models = len(parameters_list)
        cluster_labels = None

        if self.dynamic_grouping == 1:
            # Round 1: Ensure every cluster has at least one client
            if server_round == 1:
                cluster_labels = np.zeros(num_models, dtype=int)

                # Step 1: Assign the first `num_clusters` clients to different clusters
                shuffled_clients = np.random.permutation(num_models)
                for i in range(min(self.num_clusters, num_models)):  # Avoid out-of-bounds errors
                    cluster_labels[shuffled_clients[i]] = i

                # Step 2: Assign the remaining clients randomly across all clusters
                for i in range(self.num_clusters, num_models):
                    cluster_labels[shuffled_clients[i]] = np.random.randint(0, self.num_clusters)

                self.cluster_labels = cluster_labels  # Save cluster labels for subsequent rounds
                self.client_cluster_mapping = {i: cluster_labels[i] for i in range(num_models)}

            # From Round 2 onwards: Cluster based on accuracy while ensuring each cluster has at least one client
            elif server_round % self.clustering_frequency == 0:
                accuracy_scores = np.array([metrics.get('accuracy', 0) for metrics in client_metrics]).reshape(-1, 1)

                # Normalize accuracy scores for better separation
                scaler = MinMaxScaler()
                normalized_scores = scaler.fit_transform(accuracy_scores)

                kmeans = KMeans(n_clusters=self.num_clusters, random_state=0, n_init='auto')
                cluster_labels = kmeans.fit_predict(normalized_scores)

                # Ensure every cluster has at least one client after clustering
                cluster_counts = np.bincount(cluster_labels, minlength=self.num_clusters)
                missing_clusters = np.where(cluster_counts == 0)[0]

                if len(missing_clusters) > 0:
                    print(f"[Warning] Clusters {missing_clusters.tolist()} have no clients after KMeans. Adjusting assignment.")

                    # Sort clients by accuracy to distribute them evenly
                    sorted_clients = np.argsort(normalized_scores.flatten())

                    # Move clients from the largest cluster into missing clusters
                    for missing_cluster in missing_clusters:
                        # Find the most populated cluster
                        overpopulated_cluster = np.argmax(cluster_counts)
                        overpopulated_clients = np.where(cluster_labels == overpopulated_cluster)[0]

                        # Move the lowest accuracy client from that cluster to the missing cluster
                        if len(overpopulated_clients) > 1:
                            client_to_move = overpopulated_clients[0]  # Move the lowest accuracy client
                            cluster_labels[client_to_move] = missing_cluster
                            cluster_counts[overpopulated_cluster] -= 1
                            cluster_counts[missing_cluster] += 1

                # **Prevent one cluster from dominating**
                max_clients_per_cluster = num_models // self.num_clusters
                for cluster in range(self.num_clusters):
                    if cluster_counts[cluster] > max_clients_per_cluster:
                        # Move extra clients to underpopulated clusters
                        extra_clients = np.where(cluster_labels == cluster)[0][max_clients_per_cluster:]
                        underpopulated_clusters = np.where(cluster_counts < max_clients_per_cluster)[0]

                        for client in extra_clients:
                            if len(underpopulated_clusters) > 0:
                                new_cluster = underpopulated_clusters[0]
                                cluster_labels[client] = new_cluster
                                cluster_counts[cluster] -= 1
                                cluster_counts[new_cluster] += 1
                                if cluster_counts[new_cluster] >= max_clients_per_cluster:
                                    underpopulated_clusters = np.delete(underpopulated_clusters, 0)

                self.cluster_labels = cluster_labels  # Save new cluster labels
                self.client_cluster_mapping = {i: cluster_labels[i] for i in range(num_models)}

            else:
                # Use previously stored cluster labels if not a clustering round
                cluster_labels = self.cluster_labels
                if cluster_labels is None:
                    raise ValueError("Cluster labels not initialized.")

            # Aggregate parameters within each cluster
            aggregated_parameters = []
            for cluster in range(self.num_clusters):
                cluster_parameters = [parameters_list[i] for i in range(num_models) if cluster_labels[i] == cluster]
                if cluster_parameters:
                    cluster_aggregated_parameters = [np.mean(np.array(param_tuple), axis=0) for param_tuple in zip(*cluster_parameters)]
                    aggregated_parameters.append(cluster_aggregated_parameters)

            # Further aggregate cluster centers to obtain final parameters
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
        """Save the cluster assignments for each client in a single file with fixed client IDs assigned to clusters."""
        if self.dynamic_grouping != 1 or cluster_labels is None:
            return  # Skip saving if dynamic grouping is not enabled or cluster_labels is None.

        # Define the path to save cluster assignments
        cluster_assignment_file_path = os.path.join(self.results_subfolder, 'cluster_assignments.h5')
        cluster_assignment_txt_path = os.path.join(self.results_subfolder, 'cluster_assignments.txt')

        # Extract and sort client IDs numerically if possible
        client_ids = [client.cid for client, _ in results]
        try:
            client_ids = sorted(client_ids, key=lambda x: int(x))
        except ValueError:
            client_ids = sorted(client_ids)

        # Update the client-cluster mapping with fixed client assignments
        if server_round == 1 or server_round % self.clustering_frequency == 0:
            self.client_cluster_mapping = {client_id: cluster_labels[idx] for idx, client_id in enumerate(client_ids)}

        # Save the client-cluster mapping to the HDF5 file
        with h5py.File(cluster_assignment_file_path, 'a') as f:
            if str(server_round) not in f:
                grp = f.create_group(str(server_round))
                grp.create_dataset("client_ids", data=np.array(client_ids, dtype='S'))
                grp.create_dataset("cluster_labels", data=np.array([self.client_cluster_mapping[cid] for cid in client_ids], dtype='i'))
            else:
                grp = f[str(server_round)]
                grp["client_ids"][:] = np.array(client_ids, dtype='S')
                grp["cluster_labels"][:] = np.array([self.client_cluster_mapping[cid] for cid in client_ids], dtype='i')

        # Save to the TXT file with sorted entries
        with open(cluster_assignment_txt_path, 'a') as txt_file:
            txt_file.write(f"Server Round {server_round}:\n")
            for cid in client_ids:
                txt_file.write(f"Client ID: {cid}, Cluster: {self.client_cluster_mapping[cid]}\n")
            txt_file.write("\n")


    def aggregate_fit(
        self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results and save models for both dynamic and default grouping."""

        if not results:
            return None, {}

        parameters_list = [parameters_to_ndarrays(res.parameters) for client, res in results]

        # Extract client accuracy from results
        client_metrics = []
        for client, res in results:
            metrics = res.metrics
            client_metrics.append({
                "client_id": client.cid,
                "accuracy": metrics.get('accuracy', 0),  # Use accuracy for clustering
            })

        if self.dynamic_grouping == 1:
            # Perform clustering based on accuracy and aggregate parameters for each cluster
            aggregated_parameters, cluster_labels = self.aggregate_parameters(parameters_list, server_round, client_metrics)
            self.cluster_labels = cluster_labels  # Store cluster labels for this round

            # Save cluster assignments
            self._save_cluster_assignments(results, cluster_labels, server_round)
        else:
            # Default global aggregation logic
            aggregated_parameters = [np.mean(param_tuple, axis=0) for param_tuple in zip(*parameters_list)]

        aggregated_parameters_fl = fl.common.ndarrays_to_parameters(aggregated_parameters)

        return aggregated_parameters_fl, {}




    def configure_evaluate(self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.ClientManager) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """Configure evaluation with optional dynamic grouping."""
        
        if self.fraction_evaluate == 0.0:
            return []

        config = {"server_round": server_round, "task": "evaluate"}

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        evaluate_configurations = []
        for client in clients:
            client_id = int(client.cid)
            
            # Assign cluster-specific parameters to clients based on clustering
            if self.dynamic_grouping == 1 and server_round > 1:
                # Ensure client_cluster_mapping is initialized and client_id exists in the mapping
                if hasattr(self, 'client_cluster_mapping') and client_id in self.client_cluster_mapping:
                    cluster = self.client_cluster_mapping[client_id]
                    if cluster in self.cluster_models and self.cluster_models[cluster] is not None:
                        cluster_parameters = self.cluster_models[cluster]
                    else:
                        cluster_parameters = parameters  # Use global parameters if cluster model is not available
                else:
                    # If client_id is not in the mapping, use default parameters
                    cluster_parameters = parameters
            else:
                cluster_parameters = parameters

            evaluate_ins = fl.common.EvaluateIns(cluster_parameters, config=config)
            evaluate_configurations.append((client, evaluate_ins))

        return evaluate_configurations


    def log_all_clients_hardware_resources(self, server_round, client_results):
        """Log each client's hardware usage in hardware_resources.ncol and aggregate CPU/GPU for resource_consumption.txt,
        ensuring GPU usage does not exceed 100% by scaling if needed.
        """
        hardware_file_path = os.path.join(self.results_subfolder, 'hardware_resources.ncol')
        client_metrics = []
        total_gpu_usage = 0

        with open(hardware_file_path, 'a') as file:
            file.write(f"Round {server_round}\n")
            
            # First pass: Collect GPU usage
            for client, res in client_results:
                cpu_usage = round(psutil.cpu_percent(interval=1), 3)
                gpus = GPUtil.getGPUs()
                gpu_usage = round(gpus[0].load * 100, 3) if gpus else 0
                
                total_gpu_usage += gpu_usage
                client_metrics.append({
                    "client_id": client.cid,
                    "cpu": cpu_usage,
                    "gpu": gpu_usage,
                })

            # **Scale down GPU usage if total exceeds 100%**
            if total_gpu_usage > 100:
                scale_factor = 100 / total_gpu_usage  # Compute scaling factor
                for metric in client_metrics:
                    metric["gpu"] = round(metric["gpu"] * scale_factor, 3)  # Apply scaling

            # Second pass: Log adjusted results
            for metric in client_metrics:
                file.write(f"Client {metric['client_id']}: CPU {metric['cpu']}%, GPU {metric['gpu']}%\n")

        # Log aggregated resource usage after scaling
        self.log_resource_consumption(server_round, client_metrics)


    def _compute_group_metrics(self, results):
        """Compute average metrics for each group in dynamic grouping."""
        group_metrics = [{'accuracy': 0.0, 'f1_score': 0.0, 'log_loss': 0.0} for _ in range(self.num_clusters)]
        group_counts = [0] * self.num_clusters

        for client, res in results:
            cluster_idx = self.cluster_labels[int(client.cid) % len(self.cluster_labels)]
            metrics = res.metrics
            accuracy = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_score', 0)
            log_loss_value = metrics.get('log_loss', 0)
            num_examples = res.num_examples

            group_metrics[cluster_idx]['accuracy'] += accuracy * num_examples
            group_metrics[cluster_idx]['f1_score'] += f1 * num_examples
            group_metrics[cluster_idx]['log_loss'] += log_loss_value * num_examples
            group_counts[cluster_idx] += num_examples

        # Compute averages
        for idx in range(self.num_clusters):
            if group_counts[idx] > 0:
                group_metrics[idx]['accuracy'] /= group_counts[idx]
                group_metrics[idx]['f1_score'] /= group_counts[idx]
                group_metrics[idx]['log_loss'] /= group_counts[idx]

        return group_metrics


    def _select_best_model(self, server_round: int, evaluation_file_path: str) -> Tuple[int, float]:
        """Identify the best-performing cluster based on a specified evaluation metric."""
        best_cluster = None
        best_performance = -float('inf')  # Assuming higher metric is better (e.g., accuracy)
        metric_to_use = 'Accuracy'  # Change this to 'Accuracy', 'F1 Score', or 'Log Loss' as needed

        # Read the evaluation file and find the metrics for the current round
        with open(evaluation_file_path, 'r') as file:
            lines = file.readlines()

        # Locate the round's metrics in the file
        round_found = False
        for line in lines:
            if f"Round {server_round}" in line:
                round_found = True
                continue
            if round_found and line.strip().startswith("Group-"):
                # Extract group number
                match_group = re.match(r'Group-(\d+):', line)
                if match_group:
                    group = int(match_group.group(1))
                else:
                    continue  # Skip if no match

                # Extract the desired metric
                pattern = rf'{metric_to_use}:\s*([\d\.]+)'
                match_metric = re.search(pattern, line)
                if match_metric:
                    metric_value = float(match_metric.group(1))
                else:
                    continue  # Skip if the metric is not found

                # Update best cluster based on the specified metric
                if metric_value > best_performance:
                    best_performance = metric_value
                    best_cluster = group

            elif round_found and line.strip() == "":
                # End of the current round's section
                break

        if best_cluster is None:
            raise ValueError(f"No metrics found for Round {server_round} in {evaluation_file_path}")

        return best_cluster - 1, best_performance  # Adjust cluster index to match 0-based indexing


    def _select_best_model_and_save(self, server_round: int):
        """Select the best-performing model based on evaluation and save it as the global model."""
        evaluation_file_path = os.path.join(self.results_subfolder, "evaluation_loss.txt")
        
        try:
            best_cluster, best_performance = self._select_best_model(server_round, evaluation_file_path)
        except ValueError:
            print(f"[Round {server_round}] No valid cluster found for selection. Skipping best model update.")
            return

        # Ensure best_cluster exists in self.cluster_models
        if best_cluster not in self.cluster_models or self.cluster_models[best_cluster] is None:
            print(f"[Round {server_round}] Warning: Best cluster {best_cluster} not found in cluster_models. Choosing alternative.")
            
            # Find a valid cluster with trained parameters
            valid_clusters = [c for c in self.cluster_models if self.cluster_models[c] is not None]
            if not valid_clusters:
                print(f"[Round {server_round}] No valid clusters available. Skipping best model update.")
                return
            
            best_cluster = valid_clusters[0]  # Select the first valid cluster
            print(f"[Round {server_round}] Using alternative cluster {best_cluster}.")

        print(f"Best model selected from Cluster-{best_cluster} with performance: {best_performance:.4f}")

        best_model_parameters = self.cluster_models[best_cluster]

        if self.model_type == "Image Anomaly Detection":
            global_net = SparseAutoencoder()
        elif self.model_type == "Image Classification":
            global_net = MobileNetV3()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        state_dict = aggregated_parameters_to_state_dict(parameters_to_ndarrays(best_model_parameters), self.model_type)
        global_net.load_state_dict(state_dict)

        # Save the best-performing model with a specific name
        best_model_path = os.path.join(self.results_subfolder, "best_cluster_model.pth")
        torch.save(global_net.state_dict(), best_model_path)
        print(f"Best cluster model saved as {best_model_path}.")



    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results, save metrics, and select the best-performing model."""
        
        if not results:
            return None, {}

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Paths for metric files
        accuracy_file_path = os.path.join(self.results_subfolder, 'accuracy_scores.ncol')
        f1_score_file_path = os.path.join(self.results_subfolder, 'F1_scores.ncol')
        logloss_file_path = os.path.join(self.results_subfolder, 'LogLoss_scores.ncol')
        
        # Determine output file based on dynamic grouping
        evaluation_file_name = 'evaluation_loss.txt' if self.dynamic_grouping == 1 else 'aggregated_evaluation_loss.txt'
        evaluation_file_path = os.path.join(self.results_subfolder, evaluation_file_name)

        total_accuracy = 0.0
        total_f1 = 0.0
        total_logloss = 0.0
        total_examples = 0
        accuracy_scores = []
        f1_scores = []
        logloss_scores = []
        self.log_all_clients_hardware_resources(server_round, results)

        # Aggregate client metrics
        for client, res in results:
            if self.model_type == "Image Classification":
                # Get metrics
                accuracy = res.metrics.get('accuracy', 0)
                f1 = res.metrics.get('f1_score', 0)
                logloss = res.metrics.get('log_loss', 0)

                num_examples = res.num_examples

                # Append to lists
                accuracy_scores.append((client.cid, accuracy))
                f1_scores.append((client.cid, f1))
                logloss_scores.append((client.cid, logloss))

                total_accuracy += accuracy * num_examples
                total_f1 += f1 * num_examples
                total_logloss += logloss * num_examples
                total_examples += num_examples

            elif self.model_type == "Image Anomaly Detection":
                # Existing code for anomaly detection (unchanged)
                pass

        # Calculate aggregated metrics
        aggregated_accuracy = total_accuracy / total_examples if total_examples > 0 else None
        aggregated_f1 = total_f1 / total_examples if total_examples > 0 else None
        aggregated_logloss = total_logloss / total_examples if total_examples > 0 else None

        # Save client scores
        # Sort by client ID
        accuracy_scores.sort(key=lambda x: int(x[0]))
        f1_scores.sort(key=lambda x: int(x[0]))
        logloss_scores.sort(key=lambda x: int(x[0]))

        # Save accuracy scores
        with open(accuracy_file_path, 'a') as file:
            file.write(f"Time: {current_time} - Round {server_round}\n")
            for cid, metric_value in accuracy_scores:
                file.write(f"{cid} {metric_value}\n")

        # Save F1 scores
        with open(f1_score_file_path, 'a') as file:
            file.write(f"Time: {current_time} - Round {server_round}\n")
            for cid, f1_value in f1_scores:
                file.write(f"{cid} {f1_value}\n")

        # Save Log Loss scores
        with open(logloss_file_path, 'a') as file:
            file.write(f"Time: {current_time} - Round {server_round}\n")
            for cid, logloss_value in logloss_scores:
                file.write(f"{cid} {logloss_value}\n")

        # Save grouped or aggregated metrics
        with open(evaluation_file_path, 'a') as file:
            file.write(f"Time: {current_time} - Round {server_round}\n")
            if self.dynamic_grouping == 1 and self.cluster_labels is not None:
                group_metrics = self._compute_group_metrics(results)
                for group_idx, metrics in enumerate(group_metrics, start=1):
                    file.write(
                        f"Group-{group_idx}: Accuracy: {metrics['accuracy']:.4f}, "
                        f"F1 Score: {metrics['f1_score']:.4f}, Log Loss: {metrics['log_loss']:.4f}\n"
                    )
            else:
                file.write(
                    f"Aggregated Metrics: Accuracy: {aggregated_accuracy:.4f}, "
                    f"F1 Score: {aggregated_f1:.4f}, Log Loss: {aggregated_logloss:.4f}\n"
                )


        # Save the best-performing model as the global model after evaluation
        if self.dynamic_grouping == 1:
            self._select_best_model_and_save(server_round)

        return aggregated_accuracy, {}


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

    def detect_potential_poisoned_client(self, server_round: int, local_updates):
        # Load the best cluster model (only for classification)
        best_model_path = os.path.join(self.results_subfolder, "best_cluster_model.pth")
        if self.model_type != "Image Classification":
            raise ValueError("Poison detection is only supported for Image Classification.")

        # Initialize the model and load the best model's state
        best_model = MobileNetV3()
        best_model.load_state_dict(torch.load(best_model_path))
        best_last_layer = next(reversed(best_model.state_dict().values()))  # Extract the last layer of the best cluster model

        # Extract local updates and compute cosine similarity
        client_scores = {}
        for update in local_updates:
            client_id = update["client_id"]
            local_parameters = parameters_to_ndarrays(update["model"])
            local_model_state = aggregated_parameters_to_state_dict(local_parameters, self.model_type)

            # Compute cosine similarity
            local_last_layer = next(reversed(local_model_state.values()))
            similarity = cosine_similarity(best_last_layer.reshape(1, -1), local_last_layer.reshape(1, -1))[0][0]
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