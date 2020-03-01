import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import *
from inference_attack import *


def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')




os.makedirs('saves', exist_ok=True)


batch_size=64
test_batch_size=1000
epochs=75
lr=1e-3
no_cuda=False
seed=1337
log_interval=100
logfile='log.txt'
sensitivity=2  #sensitivity value that is multiplied to layer's std in order to get threshold value

# Control Seed
torch.manual_seed(seed)

# Select Device
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(seed)
else:
    print('Not using CUDA!!!')

# Loader
#kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=test_batch_size, shuffle=False)


# Define which model to use
model = LeNet(mask=True).to(device)

print(model)
print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.000)
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs):
    model.train()
    for epoch in range(epochs):

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                print(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')


def test(loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(loader.dataset)
        accuracy = 100. * correct / len(loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


# Initial training
print("--- Initial training ---")
train(epochs)
test_accuracy = test(test_loader)
train_accuracy = test(train_loader)
log(logfile, f"initial_test_accuracy {test_accuracy}")
log(logfile, f"initial_train_accuracy {train_accuracy}")
torch.save(model, f"saves/initial_model.ptmodel")
print("--- Before pruning ---")
print_nonzeros(model)

output_train, output_test, train_label, test_label = classifier_performance(model, train_loader, test_loader)
inference_accuracy=inference_via_confidence(output_train, output_test, train_label, test_label)
print("Maximum Accuracy:",inference_accuracy)

# Pruning
print("--- Starting pruning ---")
model.prune_by_std(sensitivity)
test_accuracy = test(test_loader)
train_accuracy = test(train_loader)
log(logfile, f"Test accuracy_after_pruning {test_accuracy}")
log(logfile, f"Train accuracy_after_pruning {train_accuracy}")
print("--- After pruning ---")
print_nonzeros(model)

output_train, output_test, train_label, test_label = classifier_performance(model, train_loader, test_loader)
inference_accuracy=inference_via_confidence(output_train, output_test, train_label, test_label)
print("Maximum Accuracy:",inference_accuracy)

# Retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train(epochs)
torch.save(model, f"saves/model_after_retraining.ptmodel")
test_accuracy = test(test_loader)
train_accuracy = test(train_loader)
log(logfile, f"test_accuracy_after_retraining {test_accuracy}")
log(logfile, f"train_accuracy_after_retraining {train_accuracy}")

print("--- After Retraining ---")
print_nonzeros(model)

output_train, output_test, train_label, test_label = classifier_performance(model, train_loader, test_loader)
inference_accuracy=inference_via_confidence(output_train, output_test, train_label, test_label)
print("Maximum Accuracy:",inference_accuracy)
