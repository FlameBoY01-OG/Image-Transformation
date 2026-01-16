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


