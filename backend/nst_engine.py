import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_loader(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

unloader = transforms.ToPILImage()


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    g = torch.mm(features, features.t()) 
    return g.div(a * b * c * d) 

    
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        g = gram_matrix(input)
        self.loss = nn.functional.mse_loss(g, self.target)
        return input


def total_variation_loss(img):
    batch_size, channels, height, width = img.size()
    
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    
    return (tv_h + tv_w) / (batch_size * channels * height * width)
    

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
    
def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img,
                               content_img, content_layer=['conv_4'], 
                               style_layer=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name, layer)

        if name in content_layer:
           target = model(content_img).detach()
           content_loss = ContentLoss(target)
           model.add_module("content_loss_{}".format(i), content_loss)
           content_losses.append(content_loss)

        if name in style_layer:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses
    

def run_style_transfer(content_bytes, style_bytes, num_steps=500,
                       style_weight=1000000, content_weight=1, tv_weight=10):
    
    content_img = Image.open(io.BytesIO(content_bytes)).convert('RGB')
    style_img = Image.open(io.BytesIO(style_bytes)).convert('RGB')
    
    original_size = content_img.size
    
    max_size = 768
    loader = get_image_loader(max_size)

    content_img_tensor = loader(content_img).unsqueeze(0).to(device, torch.float)
    style_img_tensor = loader(style_img).unsqueeze(0).to(device, torch.float)

    print("Loading the model...")
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)    

    model, style_losses, content_losses = get_style_model_and_losses(cnn, cnn_normalization_mean,
                                                                    cnn_normalization_std, style_img_tensor, content_img_tensor)
    
    input_img = content_img_tensor.clone()
    input_img.requires_grad_(True)

    optimizer = optim.LBFGS([input_img], max_iter=20, line_search_fn='strong_wolfe')
    print("Optimizing...")

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            
            tv_score = total_variation_loss(input_img) * tv_weight

            loss = style_score + content_score + tv_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style: {style_score.item():.2f} Content: {content_score.item():.2f} TV: {tv_score.item():.2f}")
        
            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    result_image = unloader(input_img.squeeze(0).cpu())
    
    result_image = result_image.resize(original_size, Image.LANCZOS)
    
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    
    return img_byte_arr.getvalue()