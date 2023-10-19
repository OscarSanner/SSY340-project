import torch
from torch import nn
from torchvision.models import VGG16_Weights
from torch.nn.functional import mse_loss, l1_loss
import torchvision

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.device = device

        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg.eval()
        self.vgg.to(self.device)

    def forward(self, x, y):
        pred_features = self.vgg(x)
        real_features = self.vgg(y)

        vgg_mse_l = mse_loss(pred_features, real_features)
        mse_l = mse_loss(x, y)
        l1_l = l1_loss(x,y)

        mse_scaling_factor = 0.2
        tot_l = vgg_mse_l + l1_l + (mse_scaling_factor * mse_l)
        return tot_l
