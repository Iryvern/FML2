from imports import *
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import os

# ---------------------
# ChestXrayDataset Class
# ---------------------
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png') or fname.endswith('.jpeg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('L')  # Convert image to grayscale
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image

# ----------------
# GPSDataset Class
# ----------------
class GPSDataset(Dataset):
    def __init__(self, csv_file, selected_columns=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.selected_columns = selected_columns if selected_columns else self.data.columns.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx][self.selected_columns].values.astype('float32')
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

# ----------------------
# Helper Functions
# ----------------------
def split_dataset_by_columns(num_clients, columns):
    # Calculate the number of columns per client
    columns_per_client = len(columns) // num_clients
    column_splits = []
    
    for i in range(num_clients):
        if i == num_clients - 1:
            # Last client takes the remaining columns
            column_splits.append(columns[i * columns_per_client:])
        else:
            column_splits.append(columns[i * columns_per_client:(i + 1) * columns_per_client])
    
    return column_splits

def load_datasets(num_clients: int, data_path: str, train_transform=None, test_transform=None, data_type="image"):
    if data_type == "image":
        # Load Chest X-ray dataset
        dataset = ChestXrayDataset(root_dir=data_path, transform=train_transform)
        
        total_len = len(dataset)
        train_len = int(total_len * 0.8)
        test_len = total_len - train_len
        
        trainset, testset = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))
        partition_size = train_len // num_clients
        lengths = [partition_size] * num_clients
        if train_len % num_clients != 0:
            lengths[-1] += train_len % num_clients
        datasets = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))

        trainloaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in datasets]
        testset = ChestXrayDataset(root_dir=data_path, transform=test_transform)
        testloader = DataLoader(testset, batch_size=64, shuffle=False)

    elif data_type == "gps":
        # Load GPS dataset
        data = pd.read_csv(data_path)
        columns = data.columns.tolist()
        column_splits = split_dataset_by_columns(num_clients, columns)

        # Create datasets and dataloaders for each client
        trainloaders = []
        for client_columns in column_splits:
            trainset = GPSDataset(data_path, selected_columns=client_columns, transform=train_transform)
            trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
            trainloaders.append(trainloader)

        # Create a test dataset with all columns for overall evaluation
        testset = GPSDataset(data_path, selected_columns=columns, transform=test_transform)
        testloader = DataLoader(testset, batch_size=64, shuffle=False)

    else:
        raise ValueError(f"Unsupported data_type: {data_type}. Use 'image' or 'gps'.")

    return trainloaders, testloader
