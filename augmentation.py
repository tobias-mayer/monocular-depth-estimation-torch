import torch
from torchvision import transforms
import random

class RandomHorizontalFlip(object):
    """Randomly flips the input image horizontally with a given probability.

    Args:
        probability (float): Probability of the image being flipped. Default: 0.5.
    """

    def __init__(self, probability):
        # Initializes the transform with a given probability of flipping the image horizontally
        self.probability = probability


    def __call__(self, input):
        # Applies the transform on the input image and depth maps
        image, depth = input['image'], input['depth']

        if random.random() < self.probability:
            # Flips the image and depth maps horizontally with a probability equal to self.probability
            image = torch.flip(image, dims=(2,))
            depth = torch.flip(depth, dims=(1,))

        return { 'image': image, 'depth': depth }


class RandomColorChannelSwap(object):
    """Randomly shuffles the color channels of the input image with a given probability.

    Args:
        probability (float): Probability of shuffling the color channels. Default: 0.25.
    """

    def __init__(self, probability):
        # Initializes the transform with a given probability of swapping the color channels in the image
        self.probability = probability


    def __call__(self, input):
        # Applies the transform on the input image. The depth maps are unchanged 
        image, depth = input['image'], input['depth']

        if random.random() < self.probability:
            # Swaps the color channels of the image randomly with a probability equal to self.probability
            image = image[torch.randperm(3), :, :]

        return { 'image': image, 'depth': depth }


class TensorFromPIL(object):
    """Converts a PIL image and depth map to PyTorch tensors.

    This class resizes the depth map to a fixed size of (320, 240) and converts
    both the image and depth map to PyTorch tensors.

    Args:
        None
    """

    def __call__(self, input):
        # Converts the input image and depth maps from PIL Image objects to PyTorch tensors
        image, depth = input['image'], input['depth']

        # Resizes the depth map to a fixed size of (320, 240)
        depth = depth.resize((320, 240))
        
        # Converts the image and depth map to PyTorch tensors
        image = transforms.ToTensor()(image)
        depth = transforms.ToTensor()(depth)

        return { 'image': image, 'depth': depth }


def train_transform():
    # Defines the transformation pipeline for training data
    return transforms.Compose([
        TensorFromPIL(),
        RandomHorizontalFlip(0.5),
        RandomColorChannelSwap(0.25)
    ])


def test_transform():
    # Defines the transformation pipeline for test data
    return transforms.Compose([
        TensorFromPIL()
    ])


