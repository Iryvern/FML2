from imports import *
from models import SparseAutoencoder, SimpleCNN  # Replace YOLOv11 with SimpleCNN
from training import train  # Only import the train function from training.py
import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF
import torch.optim as optim

# Set parameters for the model from a list of NumPy arrays
def set_parameters(net, parameters: list[np.ndarray], model_type: str):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    net.load_state_dict(state_dict, strict=True)


# Get parameters from the model as a list of NumPy arrays
def get_parameters(net) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, optimizer, scheduler, model_type, epochs_per_round):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
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
        return fl.common.FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Model trained"),
            parameters=updated_parameters,
            num_examples=len(self.trainloader),
            metrics={}
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")
        ndarrays = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.net, ndarrays, model_type=self.model_type)

        # Perform reconstruction and calculate SSIM using trainloader
        total_ssim = 0.0
        total_items = 0
        self.net.eval()
        with torch.no_grad():
            for batch in self.trainloader:
                images, labels = batch if self.model_type == "Image Classification" else (batch, None)
                images = images.to(DEVICE)
                outputs = self.net(images)
                
                if self.model_type == "Image Anomaly Detection":
                    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
                    outputs_np = outputs.cpu().numpy().transpose(0, 2, 3, 1)
                    
                    for img, out in zip(images_np, outputs_np):
                        img_gray = TF.to_pil_image(img).convert("L")
                        out_gray = TF.to_pil_image(out).convert("L")
                        img_gray = np.array(img_gray)
                        out_gray = np.array(out_gray)
                        ssim_value = ssim(img_gray, out_gray, data_range=img_gray.max() - img_gray.min())
                        total_ssim += ssim_value
                    total_items += images.size(0)
                else:
                    # For classification, calculate the classification loss instead of SSIM
                    labels = labels.to(DEVICE)
                    criterion = torch.nn.CrossEntropyLoss()
                    loss = criterion(outputs, labels)
                    total_ssim += loss.item()
                    total_items += images.size(0)

        average_metric = total_ssim / total_items if total_items > 0 else 0

        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Evaluation completed"),
            loss=1 - average_metric if self.model_type == "Image Anomaly Detection" else average_metric,
            num_examples=total_items,
            metrics={"ssim" if self.model_type == "Image Anomaly Detection" else "classification_loss": average_metric}
        )

# Flower client function
def client_fn(cid, trainloaders, model_type) -> FlowerClient:
    # Initialize model based on model_type
    if model_type == "Image Anomaly Detection":
        net = SparseAutoencoder().to(DEVICE)
    elif model_type == "Image Classification":
        net = SimpleCNN().to(DEVICE)  # Initialize SimpleCNN model

    trainloader = trainloaders[int(cid)]
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    return FlowerClient(cid, net, trainloader, optimizer, scheduler, model_type, epochs_per_round=3)
