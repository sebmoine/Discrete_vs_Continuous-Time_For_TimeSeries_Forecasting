import logging
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader, Dataset
from src.scripts.plot_data import show_image, show_image_per_label, show_labels_distribution, show_transformed_samples


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def get_dataloaders(config,use_cuda):
    batch_size  = config["batch_size"]
    num_workers = config["num_workers"]
    val_ratio   = config["val_ratio"]

    transform_train = transforms.Compose([
        transforms.ToTensor(),  #normalise par 255                     
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = Dataset(transform=transform_train)
    val_dataset   = Dataset(transform=transform_test)
    test_dataset  = Dataset(transform=transform_test)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size, pin_memory=use_cuda, drop_last=True, prefetch_factor = 2 if num_workers > 0 else None, persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=use_cuda, drop_last=False, prefetch_factor = 2 if num_workers > 0 else None, persistent_workers=num_workers > 0)
    test_loader  = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=use_cuda, drop_last=False, prefetch_factor = 2 if num_workers > 0 else None, persistent_workers=num_workers > 0)

    num_classes = len(train_dataset.subset.dataset.classes)
    images, _ = next(iter(train_loader))
    input_size = images.shape[1:]

    show_image(train_dataset)
    show_image_per_label(train_dataset)
    show_labels_distribution(train_dataset)
    show_transformed_samples(train_dataset)

    return train_loader, val_loader, test_loader, input_size, num_classes


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def log_loader_info(loader):
    dataset = loader.dataset
    transform = getattr(dataset, "transform", None)
    
    if transform:
        transforms_str = "\n\t\t- ".join([str(t) for t in transform.transforms])
        transforms_str = "\n\t\t- " + transforms_str
    else:
        transforms_str = "\n\t\t- None"
    
    return f"Dataset XXXXXX\n\tNumber of datapoints : {len(dataset)}\n\tTransforms:{transforms_str}"
