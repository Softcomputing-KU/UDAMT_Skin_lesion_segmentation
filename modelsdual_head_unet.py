import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50

class DualHeadSeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Shared Encoder (ResNet50)
        self.encoder = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2]
        # Main Head (FCN)
        self.main_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1)
        # Auxiliary Head (DeepLabv3+)
        self.aux_head = deeplabv3_resnet50(pretrained=True).classifier
        self.aux_head[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        main_out = self.main_head(features)
        main_out = nn.functional.interpolate(main_out, scale_factor=16, mode='bilinear')
        aux_out = self.aux_head(features)
        aux_out = nn.functional.interpolate(aux_out, scale_factor=16, mode='bilinear')
        return main_out, aux_out