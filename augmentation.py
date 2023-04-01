import torch
from torchvision import transforms
import random

class RandomHorizontalFlip(object):
    def __init__(self, probability):
        self.probability = probability


    def __call__(self, input):
        image, depth = input['image'], input['depth']

        if random.random() < self.probability:
            image = torch.flip(image, dims=(2,))
            depth = torch.flip(depth, dims=(1,))

        return { 'image': image, 'depth': depth }


class RandomColorChannelSwap(object):
    def __init__(self, probability):
        self.probability = probability


    def __call__(self, input):
        image, depth = input['image'], input['depth']

        if random.random() < self.probability:
            image = image[torch.randperm(3), :, :]

        return { 'image': image, 'depth': depth }


class TensorFromPIL(object):
    def __call__(self, input):
        print(input)
        image, depth = input['image'], input['depth']

        depth = depth.resize((320, 240))

        image = transforms.ToTensor()(image)
        depth = transforms.ToTensor()(depth)

        return { 'image': image, 'depth': depth }


def train_transform():
    return transforms.Compose([
        TensorFromPIL(),
        RandomHorizontalFlip(0.5),
        RandomColorChannelSwap(0.25)
    ])


def test_transform():
    return transforms.Compose([
        TensorFromPIL()
    ])


