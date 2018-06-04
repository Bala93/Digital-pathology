
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image


model_path  = '/media/htic/NewVolume1/murali/mitosis/weight/whole_slide.pt'
model_ft = models.resnet101()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,2)
model_ft.load_state_dict(torch.load(model_path))
model_ft.cuda()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

use_gpu = torch.cuda.is_available()
y_true = []
y_pred = []



img_path = '/media/htic/NewVolume1/murali/mitosis/dataset/scale_16_train_val/val/Tumor/A08_8_9.jpg'
img = Image.open(img_path)
img_tensor = data_transforms['val'](img).unsqueeze(0)
inputs, labels = data
if use_gpu:
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
else:
    inputs, labels = Variable(inputs), Variable(labels)

outputs = model_ft(inputs)
_, preds = torch.max(outputs.data, 1)

y_true.append(labels.data.cpu().numpwpy()) 
y_pred.append(preds.cpu().numpy())

print (y_pred[0][0],y_true[0][0])
# plt.imshow(inputs.cpu())
break