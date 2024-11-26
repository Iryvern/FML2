from imports import *
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from PIL import Image
import os
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png') or fname.endswith('.jpeg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('L')  # Convert image to grayscale for anomaly detection
        if self.transform is not None:
            image = self.transform(image)
        
        return image

class SelfDrivingCarDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Load labels from CSV file
        self.labels_df = pd.read_csv(labels_file)

        # Get unique images and corresponding class labels
        self.images = self.labels_df['frame'].unique()
        self.labels = {row['frame']: row['class_id'] for _, row in self.labels_df.iterrows()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for classification

        # Get class label for the current image
        label = self.labels[img_name]

        if self.transform is not None:
            image = self.transform(image)

        # Return the image and class label
        return image, torch.tensor(label, dtype=torch.int64)


def load_datasets(
    num_clients: int,
    dataset_path: str,
    train_transform,
    test_transform,
    model_type: str,
    data_split: List[float] = None,
):
    # Load configuration from the Default.txt file
    with open('Default.txt', 'r') as f:
        config = dict(line.strip().split('=') for line in f if '=' in line)
    
    poison_value = float(config.get('poison_percentage', 0)) / 100  # Convert to fraction (e.g., 20 -> 0.2)
    
    # Determine which dataset to use based on model_type
    if model_type == "Image Anomaly Detection":
        dataset = ChestXrayDataset(root_dir=dataset_path, transform=train_transform)
        
        # Standard train/test split for chest x-ray dataset
        total_len = len(dataset)
        train_len = int(total_len * 0.8)
        test_len = total_len - train_len
        trainset, testset = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))

    elif model_type == "Image Classification":
        images_dir = os.path.join(dataset_path, 'images')
        
        # Load dataset files
        train_labels_file = os.path.join(dataset_path, 'labels_train.csv')
        val_labels_file = os.path.join(dataset_path, 'labels_val.csv')
        
        trainset = SelfDrivingCarDataset(images_dir=images_dir, labels_file=train_labels_file, transform=train_transform)
        testset = SelfDrivingCarDataset(images_dir=images_dir, labels_file=val_labels_file, transform=test_transform)
        
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Partition training set across clients
    train_len = len(trainset)
    
    if data_split is None:
        # Default: Evenly split the data among clients
        partition_size = train_len // num_clients
        lengths = [partition_size] * num_clients
        if train_len % num_clients != 0:
            lengths[-1] += train_len % num_clients
    else:
        # Validate the custom data_split
        if sum(data_split) > 1.0:
            raise ValueError("The sum of data_split fractions must not exceed 1.0.")
        if len(data_split) != num_clients:
            raise ValueError("Length of data_split must match the number of clients.")

        # Compute the lengths for each client
        lengths = [int(train_len * fraction) for fraction in data_split]
        # Adjust for any remaining data due to rounding
        if sum(lengths) < train_len:
            lengths[-1] += train_len - sum(lengths)

    datasets = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))

    # Apply poisoning to the first client's dataset if poison_value is set
    if poison_value > 0.0:
        num_poisoned_samples = int(len(datasets[0]) * poison_value)
        print(f"Applying poisoning to {poison_value * 100:.1f}% of the data for the first client ({num_poisoned_samples} samples)...")
        datasets[0] = poison_subset(datasets[0], poison_value)

    # Print the split information for debugging
    print("Data split among clients (number of samples per client):")
    for idx, length in enumerate(lengths):
        poisoned = "*" if idx == 0 and poison_value > 0 else ""
        print(f"  Client {idx + 1}: {length} samples {poisoned}")

    # Create DataLoaders
    trainloaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in datasets]
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainloaders, testloader


def poison_subset(subset: torch.utils.data.Subset, poison_percentage: float) -> torch.utils.data.Subset:
    # Extract the dataset and indices
    dataset = subset.dataset
    indices = subset.indices

    if not isinstance(dataset, SelfDrivingCarDataset):
        raise TypeError("poison_subset currently supports only SelfDrivingCarDataset.")

    # Determine the number of rows to poison
    num_rows = len(indices)
    num_to_poison = int(num_rows * poison_percentage)

    # Shuffle indices and select the first `num_to_poison` for poisoning
    indices_to_poison = torch.randperm(num_rows)[:num_to_poison]

    # Get the minimum and maximum class_id values in the dataset
    min_class_id = dataset.labels_df['class_id'].min()
    max_class_id = dataset.labels_df['class_id'].max()

    # Apply poisoning by assigning a random value between min and max (inclusive)
    for idx in indices_to_poison:
        img_name = dataset.images[indices[idx]]
        new_class_id = torch.randint(min_class_id, max_class_id + 1, (1,)).item()  # Correct range is [min, max+1)
        dataset.labels_df.loc[dataset.labels_df['frame'] == img_name, 'class_id'] = new_class_id

    # Update the labels dictionary
    dataset.labels = {row['frame']: row['class_id'] for _, row in dataset.labels_df.iterrows()}

    return subset



