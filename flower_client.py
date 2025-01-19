from imports import *
from models import SparseAutoencoder, MobileNetV3
from training import train
import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF
import torch.optim as optim
import psutil  # To capture CPU, memory, and network stats
import GPUtil  # To capture GPU stats
from torch.optim import AdamW

# Set parameters for the model from a list of NumPy arrays
def set_parameters(net, parameters: list[np.ndarray], model_type: str):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Get parameters from the model as a list of NumPy arrays
def get_parameters(net) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, testloader, optimizer, scheduler, model_type, epochs_per_round):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader  # Added testloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_type = model_type
        self.epochs_per_round = epochs_per_round


    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")
        ndarrays = get_parameters(self.net)
        parameters = ndarrays_to_parameters(ndarrays)
        status = fl.common.Status(code=fl.common.Code.OK, message="Parameters retrieved")
        return fl.common.GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.net, ndarrays, model_type=self.model_type)
        train(self.net, self.trainloader, epochs=self.epochs_per_round, optimizer=self.optimizer, model_type=self.model_type)
        updated_ndarrays = get_parameters(self.net)
        updated_parameters = ndarrays_to_parameters(updated_ndarrays)

        # Capture hardware metrics during training
        client_resources = self._get_hardware_metrics()
        
        return fl.common.FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Model trained"),
            parameters=updated_parameters,
            num_examples=len(self.trainloader),
            metrics=client_resources  # Send the hardware metrics as part of metrics
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")
        ndarrays = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.net, ndarrays, model_type=self.model_type)

        self.net.eval()
        total_items = 0

        # Initialize lists to store labels and predictions
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in self.testloader:  # Updated to use testloader
                if self.model_type == "Image Classification":
                    images, labels = batch
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = self.net(images)

                    # Adjust outputs to include only classes 0-4
                    outputs = outputs[:, :5]  # Only keep the first 5 classes

                    # Recompute probabilities to ensure they sum to 1
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

                    total_items += images.size(0)

                elif self.model_type == "Image Anomaly Detection":
                    # Existing code for anomaly detection (unchanged)
                    pass


        # Calculate evaluation metrics
        if self.model_type == "Image Classification":
            from sklearn.metrics import f1_score, log_loss, accuracy_score

            # Define the list of labels (classes 0-4)
            label_list = [0, 1, 2, 3, 4]

            # Compute accuracy
            accuracy = accuracy_score(all_labels, all_preds)
            # Compute F1 score (weighted average), set zero_division=0
            f1 = f1_score(all_labels, all_preds, average='weighted', labels=label_list, zero_division=0)
            # Compute Log Loss
            logloss = log_loss(all_labels, all_probs, labels=label_list)

            # Prepare metrics to return
            metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "log_loss": logloss
            }
            loss = 1 - accuracy  # You can adjust this based on your loss function

        elif self.model_type == "Image Anomaly Detection":
            # Existing code for anomaly detection (unchanged)
            pass

        # Capture hardware metrics during evaluation
        client_resources = self._get_hardware_metrics()

        # Combine metrics
        metrics.update(client_resources)

        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Evaluation completed"),
            loss=loss,
            num_examples=total_items,
            metrics=metrics
        )



    def _get_hardware_metrics(self):
        """Collect hardware metrics on the client's machine."""
        # Capture CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)

        # Capture GPU usage
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0

        # Capture Memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent

        # Capture Network usage
        net_io = psutil.net_io_counters()
        net_sent = round(net_io.bytes_sent / (1024 ** 2), 2)  # Convert to MB
        net_received = round(net_io.bytes_recv / (1024 ** 2), 2)  # Convert to MB

        return {
            "cpu_usage": cpu_usage,
            "gpu_usage": gpu_usage,
            "memory_usage": memory_usage,
            "network_sent": net_sent,
            "network_received": net_received,
        }

# Flower client function
def client_fn(cid, trainloaders, testloaders, model_type) -> FlowerClient:
    # Initialize model based on model_type
    if model_type == "Image Anomaly Detection":
        net = SparseAutoencoder().to(DEVICE)
    elif model_type == "Image Classification":
        net = MobileNetV3().to(DEVICE)

    trainloader = trainloaders[int(cid)]
    testloader = testloaders[int(cid)]

    # Update optimizer to AdamW with weight decay
    optimizer = optim.AdamW(net.parameters(), lr=0.0001, weight_decay=1e-4)

    # Update scheduler to CosineAnnealingWarmRestarts for dynamic learning rate adjustment
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

    # Return the FlowerClient with the updated optimizer and scheduler
    return FlowerClient(cid, net, trainloader, testloader, optimizer, scheduler, model_type, epochs_per_round=3)
