from imports import *
from models import SparseAutoencoder, GRUAnomalyDetector
from training import train  # Import the train function from training.py
import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF

# Set parameters for the model from a list of NumPy arrays
def set_parameters(net, parameters: list[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Get parameters from the model as a list of NumPy arrays
def get_parameters(net) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, optimizer, scheduler, epochs_per_round, data_type="image"):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_per_round = epochs_per_round
        self.data_type = data_type

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        ndarrays = get_parameters(self.net)
        parameters = ndarrays_to_parameters(ndarrays)
        status = fl.common.Status(code=fl.common.Code.OK, message="Parameters retrieved")
        return fl.common.GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.net, ndarrays)
        train(self.net, self.trainloader, epochs=self.epochs_per_round, optimizer=self.optimizer, data_type=self.data_type)
        parameters = get_parameters(self.net)
        parameters_msg = ndarrays_to_parameters(parameters)
        status = fl.common.Status(code=fl.common.Code.OK, message="Training completed")
        return fl.common.FitRes(status=status, parameters=parameters_msg, num_examples=len(self.trainloader.dataset))

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.net, ndarrays)
        self.net.eval()
        with torch.no_grad():
            for batch in self.trainloader:
                inputs = batch.float().to(DEVICE)
                outputs = self.net(inputs)
                ssim_value = torch.nn.functional.mse_loss(outputs, inputs).item()
        return fl.common.EvaluateRes(status=fl.common.Status(code=fl.common.Code.OK, message="Evaluation completed"), loss=1 - ssim_value, num_examples=len(self.trainloader.dataset))

def client_fn(cid, trainloaders, data_type="image") -> FlowerClient:
    if data_type == "image":
        net = SparseAutoencoder().to(DEVICE)
    elif data_type in ["gps", "time_series_anomaly_detection"]:
        input_size = 6
        hidden_size = 128
        num_layers = 2
        output_size = 6
        net = GRUAnomalyDetector(input_size, hidden_size, num_layers, output_size).to(DEVICE)
    trainloader = trainloaders[int(cid)]
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    return FlowerClient(cid, net, trainloader, optimizer, scheduler, epochs_per_round=3, data_type=data_type)