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

    
class StyleLoss(nn.Module):
    def __init__(self,target_feature):
        super(StyleLoss,self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self,input):
        g = gram_matrix(input)
        self.loss = nn.functional.mse_loss(g,self.target)
        return input
    
class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization,self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)

    def forward(self,img):
        return (img-self.mean)/self.std
    
def get_style_model_and_losses(cnn,normalizatio_mean,normalization_std,style_img,
                               content_img,content_layer=['conv_4'],style_layer=['conv_1','conv_2','conv_3','conv_4','conv_5']):
    
    normalization = Normalization(normalizatio_mean,normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i=0
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            i+=1
            name = "conv_{}".format(i)
        elif isinstance(layer,nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer,nn.MaxPool2d):
            name = "pool_{}".format
        elif isinstance(layer,nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name,layer)

        if name in content_layer:
           target = model(content_img).detach()
           content_loss = ContentLoss(target)
           model.add_module("content_loss{}".format(i),content_loss)
           content_losses.append(content_loss)

        if name in style_layer:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_model("style_loss_{}".format(i),style_loss)
            style_losses.append(style_loss)

        for i in range(len(model) -1,-1,-1):
            if isinstance(model[i],ContentLoss) or isinstance(model[i],StyleLoss):
                break

        model = model[:(i+1)]

        return model,style_losses,content_losses
    


