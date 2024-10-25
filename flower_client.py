from imports import *
from models import SparseAutoencoder, YOLOv11
from training import train  # Only import the train function from training.py
import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF

# Set parameters for the model from a list of NumPy arrays
def set_parameters(net, parameters: list[np.ndarray], model_type="autoencoder"):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Get parameters from the model as a list of NumPy arrays
def get_parameters(net, model_type="autoencoder") -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class FlowerClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, optimizer, scheduler, epochs_per_round, model_type="autoencoder"):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_per_round = epochs_per_round
        self.model_type = model_type

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")
        ndarrays = get_parameters(self.net, model_type=self.model_type)
        parameters = ndarrays_to_parameters(ndarrays)
        status = fl.common.Status(code=fl.common.Code.OK, message="Parameters retrieved")
        return fl.common.GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        print(f"[Client {self.cid}] fit, config: {ins.config}")
        ndarrays = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.net, ndarrays, model_type=self.model_type)
        train(self.net, self.trainloader, epochs=self.epochs_per_round, optimizer=self.optimizer)  # Pass optimizer
        updated_ndarrays = get_parameters(self.net, model_type=self.model_type)
        updated_parameters = ndarrays_to_parameters(updated_ndarrays)
        status = fl.common.Status(code=fl.common.Code.OK, message="Model trained")
        return fl.common.FitRes(status=status, parameters=updated_parameters, num_examples=len(self.trainloader), metrics={})

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")
        ndarrays = parameters_to_ndarrays(ins.parameters)
        set_parameters(self.net, ndarrays, model_type=self.model_type)

        # Perform reconstruction and calculate SSIM using trainloader
        total_ssim = 0.0
        total_items = 0
        self.net.eval()
        with torch.no_grad():
            for images in self.trainloader:
                images = images.to(DEVICE)
                outputs = self.net(images)
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

        average_ssim = total_ssim / total_items if total_items > 0 else 0

        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Evaluation completed"),
            loss=1 - average_ssim,  # Loss is 1 - SSIM to keep lower loss better
            num_examples=total_items,
            metrics={"ssim": average_ssim}
        )

# Flower client function
def client_fn(cid, trainloaders, model_type="autoencoder") -> FlowerClient:
    if model_type == "autoencoder":
        net = SparseAutoencoder().to(DEVICE)
    elif model_type == "yolo":
        net = YOLOv11().to(DEVICE)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    trainloader = trainloaders[int(cid)]
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    return FlowerClient(cid, net, trainloader, optimizer, scheduler, epochs_per_round=3, model_type=model_type)
