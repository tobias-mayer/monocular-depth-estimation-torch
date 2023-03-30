import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

CONV_KERNEL_SIZE = 3
CONV_STRIDE = 1
CONV_PADDING = 1
RELU_SLOPE = 0.2

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.pretrained_classifier = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)

    def forward(self, x):
        features = [x]

        for k, v in self.pretrained_classifier.features._modules.items():
            features.append(v(features[-1]))

        # return intermediate layer results as well for skip connections
        return features[3], features[4], features[6], features[8], features[12]


class Upscale(nn.Sequential):
    def __init__(self, input_features, output_features):
        super(Upscale, self).__init__()        
        self.convA = nn.Conv2d(input_features, output_features, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.leakyreluA = nn.LeakyReLU(RELU_SLOPE)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=CONV_KERNEL_SIZE, stride=CONV_STRIDE, padding=CONV_PADDING)
        self.leakyreluB = nn.LeakyReLU(RELU_SLOPE)

    def forward(self, x0, x1):
        # upscale the feature map to match the output dimensions of the corresponding layer in the encoder (=skip connection input)
        upscaled_x0 = F.interpolate(x0, size=[x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.convA(torch.cat([upscaled_x0, x1], dim=1))))


class Decoder(nn.Module):
    def __init__(self, num_features=1664):
        super(Decoder, self).__init__()
        features = num_features

        self.convA = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.upA = Upscale(features // 1 + 256, features // 2)
        self.upB = Upscale(features // 2 + 128, features // 4)
        self.upC = Upscale(features // 4 + 64, features // 8)
        self.upD = Upscale(features // 8 + 64, features // 16)

        self.convB = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        original_input, x_skip1, x_skip2, x_skip3, x_skip4 = features
        x_d0 = self.convA(F.relu(x_skip4))

        x_d1 = self.upA(x_d0, x_skip3)
        x_d2 = self.upB(x_d1, x_skip2)
        x_d3 = self.upC(x_d2, x_skip1)
        x_d4 = self.upD(x_d3, original_input)
        return self.convB(x_d4)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


