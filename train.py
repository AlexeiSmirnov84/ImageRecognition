# Imports here
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import random

import os
import glob
import pandas as pd

import matplotlib.pyplot as plt 

import argparse
import configparser
import json


parser = argparse.ArgumentParser(description='New inputs:')
parser.add_argument('data_dir', action="store", default='ImageClassifier/flowers')
parser.add_argument('--save_dir', default='')
parser.add_argument('--arch', default='vgg16')
parser.add_argument('--learning_rate', default='0.001')
parser.add_argument('--hidden_units', default='500')
parser.add_argument('--epochs', default='3')
parser.add_argument('--gpu', default='cuda')

command_line = parser.parse_args()
print("data_dir: {}".format(command_line.data_dir))
print("save_dir: {}".format(command_line.save_dir))
data_dir = command_line.data_dir
save_dir = str(command_line.save_dir) + '/checkpoint.pth'

arch = str(command_line.arch)
arch_options = ['vgg13', 'vgg16']
if arch not in arch_options: print("Architecture {} is not supported. Please make your choice from: {}.".format(arch, arch_options))

learning_rate = float(command_line.learning_rate)
hidden_units = int(command_line.hidden_units)
epochs = int(command_line.epochs)
gpu = str(command_line.gpu)
if gpu and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



# TODO: Define your transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.CenterCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

valid_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)





    
model = getattr(models, arch)(pretrained=True)

input_units = 25088


# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_units, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier




criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

print_every = 40
steps = 0

model.to(device)
  
for e in range(epochs):
    running_loss = 0
    test_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
        
            with torch.no_grad():
                for inputs , labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    loss = criterion(logps,labels)
                    test_loss += loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.cuda.FloatTensor)).item()

            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation loss: {:.4f}".format(test_loss/len(validloader)),
                  "Validation Accuracy {:.4f}".format(accuracy/len(validloader)*100))
                   
            running_loss = 0
            
            
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        
        #images, labels = data
        images, labels = inputs.to(device), labels.to(device)
        
        #outputs = model(images)
        outputs = model.forward(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))




correct = 0
total = 0
with torch.no_grad():
    for data in validloader:
        
        #images, labels = data
        images, labels = inputs.to(device), labels.to(device)
        
        #outputs = model(images)
        outputs = model.forward(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 validation images: %d %%' % (100 * correct / total))



# TODO: Save the checkpoint 
model.class_to_idx = train_dataset.class_to_idx

checkpoint = {'classifier' : model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir)