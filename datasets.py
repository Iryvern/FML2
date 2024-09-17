from imports import *

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png') or fname.endswith('.jpeg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

def load_datasets(num_clients: int, dataset_path: str, train_transform, test_transform):
    # Initialize dataset
    dataset = ChestXrayDataset(root_dir=dataset_path, transform=train_transform)
    
    # Calculate lengths for training and testing
    total_len = len(dataset)
    train_len = int(total_len * 0.8)  # 80% for training
    test_len = total_len - train_len  # remaining for testing
    
    # Split dataset into training and testing
    trainset, testset = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))

    # Split training set into `num_clients` partitions
    partition_size = train_len // num_clients
    lengths = [partition_size] * num_clients
    if train_len % num_clients != 0:
        lengths[-1] += train_len % num_clients
    datasets = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))

    # Create DataLoaders for each client's dataset
    trainloaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in datasets]

    # Prepare the test set DataLoader with the appropriate transformation
    testset = ChestXrayDataset(root_dir=dataset_path, transform=test_transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainloaders, testloader
