import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import models, datasets

class Resnet(nn.Module):
   
    def __init__(self, output_dim):
       
        super(Resnet, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        
        classes = ['bacterial', 'normal', 'virus']
        in_f = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features = in_f, out_features = len(classes), bias = True)
    
        self.model = resnet
        
    def forward(self, x):
        x = self.model(x)
        return x