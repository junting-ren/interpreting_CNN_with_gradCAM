import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
from random import shuffle
from PIL import Image

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

    
    

class resnet(torch.nn.Module):
    def __init__(self,pretrained = True):
        super(resnet, self).__init__()
        
        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        res = torchvision.models.resnet18(weights = weights)
        self.features_conv  =  torch.nn.Sequential(*list(res.children())[:-2])
        #import pdb; pdb.set_trace()
        self.features_conv.avgpool = Identity()
        self.features_conv.fc = Identity()
        
        # get the avg pool of the features stem
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # get the classifier of the resnet
        self.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
        
        # softmax function
        self.softmax = torch.nn.Softmax(dim=1)
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        #import pdb; pdb.set_trace()
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.pool(x)
        x = self.fc(torch.squeeze(x))
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    
    def pred_prob(self,x):
        x = self.forward(x)
        return self.softmax(x)
    
def get_model(config_data, device):
    if config_data['model_type'] == 'pretrained':
        pretrained = True
    else:
        pretrained = False
    model = resnet(pretrained)
    return model.to(device)