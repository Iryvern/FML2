from datasets import load_datasets
from flower_client import FlowerClient
from strategy import FedCustom
from imports import *
from models import SparseAutoencoder

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize for 1 channel
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize for 1 channel
])

# Load dataset
train_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

NUM_CLIENTS = 10
trainloaders, testloader = load_datasets(NUM_CLIENTS, 'NewNormal', train_transform, test_transform)

def client_fn(cid) -> FlowerClient:
    net = SparseAutoencoder().to(DEVICE)  # Assuming SparseAutoencoder is the model class
    trainloader = trainloaders[int(cid)]
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Set your desired learning rate here
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    return FlowerClient(cid, net, trainloader, optimizer, scheduler, epochs_per_round=3)  # Adjust number of epochs per client per round

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=1000),
    strategy=FedCustom(),
    client_resources={"num_cpus": 1, "num_gpus": 0.05} ,
)
