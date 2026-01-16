import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import models
from PIL import Image
import io

device  = torch.device("cuds" if torch.cuda.is_available() else "cpu")

img_size = 500
loader = transforms.Compose([
    transforms.Resize(img_size,img_size),
    transforms.ToTensor(),
])

unloader = transforms.ToPILImage()


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input,self.target)
        return input
    

# Gram Matrix to calculate the correlation btw the feture maps
def gram_matrix(input):
    a,b,c,d = input.size()
    features = input.view(a*b,c*d)
    g = torch.mm(features,features.t()) # matrix multiplicati0n
    return g.div(a*b*c*d) # normalize

    
