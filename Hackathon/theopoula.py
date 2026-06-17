import os
import argparse
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from THEO_POULA.optimizers import Theopoula

def get_parser():
    parser = argparse.ArgumentParser(description='INI Hackathon')
    parser.add_argument('--epochs', default=40, type=int, help='Total number of training epochs')
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--model', default='mobile', type=str, help='model')
    parser.add_argument('--optim', default='theopoula', type=str, help='optimizer')
    parser.add_argument('--lr_max', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta', default=1e12, type=float)
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizers')
    
    # add hyperparameters you need
    
    return parser

device = torch.device('cuda')
# Define the data transformations
data_dir = './food-101'

parser = get_parser()
args = parser.parse_args()

print(args.model)
print(args.optim)


# Define the transformations for the training and test sets
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(17),
    transforms.RandomRotation(47),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
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
train_dataset = datasets.Food101(root=data_dir, split='train', download=False, transform=transform_train)
test_dataset = datasets.Food101(root=data_dir, split='test', download=False, transform=transform_test)

# Define the DataLoader for training and test sets
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

if args.model == 'squeeze':
    model = models.squeezenet1_0(weights=None)
    num_ftrs = model.classifier[1].in_channels
    model.classifier[1] = nn.Conv2d(num_ftrs, 101, kernel_size=(1,1), stride=(1,1))
    model.classifier[1] = nn.Conv2d(num_ftrs, 101, kernel_size=(1,1), stride=(1,1))
    model = model.to(device)
elif args.model == 'mobile':
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 101)
    model = model.to(device)

if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'adam':
    optimizer = optim.AdamW(model.parameters(), args.lr_max, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
elif args.optim == 'theopoula':
    optimizer = Theopoula.THEOPOULA(model.parameters(),eps=1e-1, lr=args.lr_max,weight_decay=args.weight_decay, beta=1e12, eta=5e-4, r=0)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)
n_epochs = args.epochs

criterion = nn.CrossEntropyLoss()

loss_list = []

epoch_number = 0
best_loss = 1000000000
best_accuracy = 0

for epoch in range(n_epochs):
    print(epoch)
    loss_train = 0
    epoch_numeber = 0.
    
    for x,y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss_train = criterion(outputs, y)
        loss_train.backward()
        optimizer.step()

    
    if epoch_number  % 5 ==0:
        print('epoch_number: {}, current_loss: {}, best_loss: {}'.format(epoch_number, loss_train, best_loss))
        
    if loss_train < best_loss:
        best_loss = loss_train

        model_path = 'ckpts/theopoula/model_best.pth'
        torch.save(model.state_dict(), model_path)

    loss_list.append(loss_train)
    epoch_number += 1

model_path = 'ckpts/theopoula/model_last.pth'
torch.save(model.state_dict(), model_path)

