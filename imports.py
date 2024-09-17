import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
import os
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy
from torch.optim.lr_scheduler import ReduceLROnPlateau

DEVICE = torch.device("cuda")  #"cuda" or "cpu"