import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define the data directory
data_dir = './food-101'

# Define the transformations for the training and test sets
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download and prepare the Food-101 dataset
train_dataset = datasets.Food101(root=data_dir, split='train', download=True, transform=transform_train)
test_dataset = datasets.Food101(root=data_dir, split='test', download=True, transform=transform_test)

# Define the DataLoader for training and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print('finish')