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
parser.add_argument('input', action="store", default='checkpoint.pth')
parser.add_argument('--top_k', default='5')
parser.add_argument('--category_names', default='ImageClassifier/cat_to_name.json')
parser.add_argument('--gpu', default='cuda')
parser.add_argument('--image', default='None')

command_line = parser.parse_args()
topk = int(command_line.top_k)

if command_line.image == "None": im = random.choice(glob.glob('ImageClassifier/flowers/test/*/*.jpg'))
else: im = command_line.image


with open(command_line.category_names, 'r') as f:
    cat_to_name = json.load(f)

model = models.densenet121(pretrained=True)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model



model = load_checkpoint(command_line.input)
model.to(command_line.gpu)


def process_image(image):
    im = Image.open(image)
    print(im)
    
    width, height = im.size
    if width > height:
        width = int(width / height * 256)
        height = 256
    else:
        height = int(height / width * 256)
        width = 256    
    im = im.resize([width,height])
    
    cropto = 224
    crop_width = (width-cropto)/2
    left = int(crop_width)
    right = int(width-crop_width)
    crop_height = (height-cropto)/2
    top = int(crop_height)
    bottom = int(height-crop_height)
    im = im.crop((left,top,right,bottom))
    print(im)
    
    np_im = np.array(im) / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_im=(np_im - mean) / std
    np_im=np_im.transpose((2,0,1))
    return np_im
    

process_image(im)

print(process_image)



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



#def predict(image_path, model, topk=5):
#    Predict the class (or classes) of an image using a trained deep learning model.
    
    
    # TODO: Implement the code to predict the class from an image file
    
def predict(image_path, model, topk):

    model.eval()
    
    processed_img=torch.from_numpy(process_image(image_path)).type(torch.FloatTensor).unsqueeze_(0)

    with torch.no_grad():
        processed_img=processed_img.to(command_line.gpu)
        
        output=model.forward(processed_img)
                
        probs, labels = torch.topk(output.data,topk)
        
        top_prob = probs.exp()
                
    class_idx_dict = {model.class_to_idx[key]: key for key in model.class_to_idx}
    
    classes = list()
    cpu_labels = labels.cpu()
    for label in cpu_labels.detach().numpy()[0]:
        classes.append(class_idx_dict[label])
        
    return top_prob.cpu().numpy()[0],classes


top_probs, top_classes = predict(im, model, topk)  

print(im)
print(top_probs)
print(top_classes)



# TODO: Display an image along with the top 5 classes
class_names = [cat_to_name[i] for i in top_classes]
names = np.array(class_names)
values = np.array(top_probs)
dataset = pd.DataFrame({'names': list(names), 'values': values}, columns=['names', 'values'])
dataset = dataset.sort_values('values')


