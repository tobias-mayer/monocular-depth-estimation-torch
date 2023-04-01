import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

CONV_KERNEL_SIZE = 3
CONV_STRIDE = 1
CONV_PADDING = 1
RELU_SLOPE = 0.2

class Encoder(nn.Module):
    """
    Encoder class that loads a pre-trained DenseNet-169 model and returns intermediate features for skip connections.
    
    Args:
        None
    
    Attributes:
        pretrained_classifier (nn.Module): Pre-trained DenseNet-169 model loaded from torchvision.models.
    
    Methods:
        forward(x): Takes input tensor x and returns intermediate feature maps for skip connections.
    """
    def __init__(self):
        super(Encoder, self).__init__()       
        self.pretrained_classifier = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)

    def forward(self, x):
        """
        Takes input tensor x and returns intermediate feature maps for skip connections.
        
        Args:
            x (torch.Tensor): Input tensor to the model.
        
        Returns:
            tuple: A tuple of intermediate feature maps from the encoder, as follows:
                - features[3] (torch.Tensor): The output from the 3rd layer of the DenseNet-169 model.
                - features[4] (torch.Tensor): The output from the 4th layer of the DenseNet-169 model.
                - features[6] (torch.Tensor): The output from the 6th layer of the DenseNet-169 model.
                - features[8] (torch.Tensor): The output from the 8th layer of the DenseNet-169 model.
                - features[12] (torch.Tensor): The output from the 12th layer of the DenseNet-169 model.
        """
        features = [x]

        for k, v in self.pretrained_classifier.features._modules.items():
            features.append(v(features[-1]))

        # return intermediate layer results as well for skip connections
        return features[3], features[4], features[6], features[8], features[12]


class Upscale(nn.Sequential):
    """
    Upscale block that upsamples the feature map to match the output dimensions of the corresponding layer in the encoder (=skip connection input).
    
    Args:
        input_features (int): Number of input features to the upscale block.
        output_features (int): Number of output features from the upscale block.
    
    Attributes:
        convA (nn.Conv2d): 2D convolutional layer applied to the input feature map.
        leakyreluA (nn.LeakyReLU): Leaky ReLU activation applied to the output of convA.
        convB (nn.Conv2d): 2D convolutional layer applied to the output of leakyreluA.
        leakyreluB (nn.LeakyReLU): Leaky ReLU activation applied to the output of convB.
    
    Methods:
        forward(x0, x1): Takes two input tensors x0 and x1, and returns the output tensor after applying the upscale block.
    """
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
    """
    A decoder module that reconstructs an image from features extracted by the encoder.

    Args:
        num_features (int, optional): Number of input features. Defaults to 1664.

    Attributes:
        convA (nn.Conv2d): Convolutional layer for initial processing of input features.
        upA (Upscale): Upscaling module A.
        upB (Upscale): Upscaling module B.
        upC (Upscale): Upscaling module C.
        upD (Upscale): Upscaling module D.
        convB (nn.Conv2d): Convolutional layer for output processing.

    Methods:
        forward(features): Forward pass of the decoder.

    """
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
    """
    A class representing the full depth estimation model that consists of an encoder and a decoder.

    ...

    Attributes
    ----------
    encoder : Encoder
        The encoder object that converts the input image into feature maps.
    decoder : Decoder
        The decoder object that constructs the output depth map from the feature maps.

    Methods
    -------
    forward(x)
        Passes the input image through the encoder and then through the decoder, 
        returning the corresponding depth map.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


