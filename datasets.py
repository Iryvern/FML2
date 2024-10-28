from imports import *
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from PIL import Image
import os

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

class SelfDrivingCarDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Load labels from CSV file
        self.labels_df = pd.read_csv(labels_file)

        # Group labels by image filename
        self.image_groups = self.labels_df.groupby('frame')
        self.images = self.image_groups.groups.keys()  # Unique image names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = list(self.images)[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # Convert to RGB

        # Get all bounding boxes and class_ids for this image
        boxes = []
        labels = []
        for _, row in self.image_groups.get_group(img_name).iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            class_id = row['class_id']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)

        if self.transform is not None:
            image = self.transform(image)
        
        # Package bounding boxes and labels as a dictionary
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        
        return image, target

def load_datasets(num_clients: int, dataset_path: str, train_transform, test_transform, model_type: str):
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
        
        # Use separate CSV files for train and validation
        train_labels_file = os.path.join(dataset_path, 'labels_train.csv')
        val_labels_file = os.path.join(dataset_path, 'labels_val.csv')
        
        trainset = SelfDrivingCarDataset(images_dir=images_dir, labels_file=train_labels_file, transform=train_transform)
        testset = SelfDrivingCarDataset(images_dir=images_dir, labels_file=val_labels_file, transform=test_transform)
        
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Partition training set across clients
    train_len = len(trainset)
    partition_size = train_len // num_clients
    lengths = [partition_size] * num_clients
    if train_len % num_clients != 0:
        lengths[-1] += train_len % num_clients

    datasets = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))
    trainloaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in datasets]
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainloaders, testloader
