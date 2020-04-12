import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import models, datasets

class VGG19(nn.Module):
   
    def __init__(self, output_dim):
       
        super(VGG19, self).__init__()
        vgg19 = torchvision.models.vgg19_bn(pretrained=True)
        
        classes = ['bacterial', 'normal', 'virus']
        in_f = vgg19.classifier[6].out_features
        vgg19.classifier.add_module(module = nn.ReLU(inplace=True), name = '7')
        vgg19.classifier.add_module(module = nn.Dropout(p=0.5, inplace=False), name='8')
        vgg19.classifier.add_module(module= nn.Linear(in_features = in_f, out_features = len(classes), bias=True), name='9')
        
    
        self.model = vgg19

    def forward(self, x):
        x = self.model(x)
        return x
    
        