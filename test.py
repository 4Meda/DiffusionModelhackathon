import os
import argparse
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='INI Hackathon')
    parser.add_argument('--ckpt', type=str, help='checkpoint')  
    # add hyperparameters you need
    
    return parser

parser = get_parser()
args = parser.parse_args()

device = torch.device('cuda')
data_dir = './food-101'

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download and prepare the Food-101 dataset
test_dataset = datasets.Food101(root=data_dir, split='test', download=False, transform=transform_test)

# Define the DataLoader for training and test sets
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

pred_list = []
label_list = []

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 101)
model = model.to(device)

for x,y in test_loader:
    x = x.to(device)
    y = y.to(device)
    
    label_list.extend(y.detach().cpu().numpy())

    ckpt_path = '/users/yx2022/sharedscratch/projects/hakcathon/ckpts/adamw/' + args.ckpt
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    logits = model(x)
    probabilities = nn.functional.softmax(logits, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)

    pred_list.extend(predicted_classes.detach().cpu().numpy().tolist())


label_list = np.array(label_list)
pred_list = np.array(pred_list)

np.save('label.npy', label_list)
np.save('pred.npy', pred_list)
