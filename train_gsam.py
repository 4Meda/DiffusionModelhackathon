import os
import argparse
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
import torch.nn as nn

from THEO_POULA.optimizers import Theopoula
 
# from GSAM.model.wide_res_net import WideResNet
from GSAM.model.smooth_cross_entropy import smooth_crossentropy
from GSAM.utility.log import Log
from GSAM.utility.initialize import initialize
from GSAM.utility.step_lr import StepLR
from GSAM.utility.bypass_bn import enable_running_stats, disable_running_stats
 
import sys; sys.path.append("..")
from GSAM.gsam import sam
from GSAM.gsam import GSAM, LinearScheduler, CosineScheduler, ProportionScheduler
 
 

def get_parser():
    parser = argparse.ArgumentParser(description='INI Hackathon')
    parser.add_argument('--total_epoch', default=1, type=int, help='Total number of training epochs')
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--model', default='squeeze', type=str, help='model')
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr_max', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_min', default=0, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument('--beta', default=1e12, type=float)
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--batchsize', type=int, default=128, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay for optimizers')
    parser.add_argument("--alpha", default=0.2, type=float, help="Alpha parameter for SAM.")
    parser.add_argument("--rho_max", default=2.0, type=float, help="Rho max parameter for SAM.")
    parser.add_argument("--rho_min", default=2.0, type=float, help="Rho min parameter for SAM.")
    
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
train_dataset = datasets.Food101(root=data_dir, split='train', download=False, transform=transform_train)
test_dataset = datasets.Food101(root=data_dir, split='test', download=False, transform=transform_test)

# Define the DataLoader for training and test sets
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)


if args.model == 'squeeze':
    model = models.squeezenet1_0(weights=None)
    num_ftrs = model.classifier[1].in_channels
    model.classifier[1] = nn.Conv2d(num_ftrs, 101, kernel_size=(1,1), stride=(1,1))
    model = model.to(device)
elif args.model == 'mobile':
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 101)
    model = model.to(device)

if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)
elif args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)
elif args.optim == 'gsam':
    # base_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # base_optimizer = Theopoula.THEOPOULA(model.parameters(),eps=1e-8, lr=args.lr,weight_decay=args.weight_decay, beta=1e12, eta=0, r=0)
    base_optimizer = optim.AdamW(model.parameters(), args.lr_max, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    #scheduler = StepLR(base_optimizer, args.learning_rate, args.epochs)
    #rho_scheduler = LinearScheduler(T_max=args.epochs*len(dataset.train), max_value=args.rho_max, min_value=args.rho_min)
    
    scheduler = CosineScheduler(T_max=args.total_epoch*75750, max_value=args.lr_max, min_value=0, optimizer=base_optimizer)
    rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.lr_max, min_lr=args.lr_min, max_value=args.rho_max, min_value=args.rho_min)
    
    optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)

num = args.batchsize*50
log = Log(log_each=num)
for epoch in range(args.total_epoch):
        model.train()
        log.train(len_dataset=75750)
 
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            '''
            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)
 
            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)
            '''
            def loss_fn(predictions, targets):
                return smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing).mean()
 
            optimizer.set_closure(loss_fn, inputs, targets)
            predictions, loss = optimizer.step()
    
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu().repeat(args.batchsize), correct.cpu(), scheduler.lr())
                scheduler.step()
                optimizer.update_rho_t()
 
        model.eval()
        log.eval(len_dataset=25250)
 
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
 
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

log.flush()
 
model_path = 'ckpts/model_last.pth'
torch.save(model.state_dict(), model_path)