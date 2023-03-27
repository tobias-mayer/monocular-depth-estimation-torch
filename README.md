# High Quality Monocular Depth Estimation via Transfer Learning

PyTorch implementation of I. Alhashim and P. Wonka, “High Quality Monocular Depth Estimation via Transfer Learning,” CoRR, vol. abs/1812.11941, 2018, [LINK](https://arxiv.org/pdf/1812.11941.pdf)

The model is used to create a 3D map of a robot's surroundings, enabling it to detect and avoid obstacles in real-time.

## How it works

- The paper introduces a transfer learning approach to monocular depth estimation. A pretrained image encoder is used that was originally designed for image classification. Encoders, that do not downsample the spatial resolution of the input tend to produce sharper depth estimations.
- Encoder is a pretrained DenseNet-169
- Decoder consists of multiple Blocks of 2x bilinear upsampling followed by two standard convolutional layers
- Loss: balances between reconstructing depth images by minimizing the difference of the depth values while also penalizing distortions of igh frequency details in the image domain of the depth map -> These details typically correspond to the boundaries of objects in the scene.
The loss is defined as a weighted sum of three loss functions.
- Loss problem: loss is larger for bigger ground-truth values. 
  Solution: reciprocal of the depth is used. -> target depth map y = m / y_(orig) where m is the max depth in the scene
- Augmentation: horizontal flip with 0.5 probability, swapping color channels (e.g. red <-> green) on the input image with 0.25 probability

Training:
- DenseNet-169 pretrained on ImageNet
- Decoder uses random initialization
- Adam optimizer with lr=0.0001 beta1=0.9, beta2=0.999
- batch size = 8
- ~42.6M trainable parameters
