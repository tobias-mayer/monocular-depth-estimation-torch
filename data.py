import os.path
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from augmentation import train_transform, test_transform

class MemoryDataset(Dataset):
    def __init__(self, csv_path='./dataset/nyu2_train.csv', img_dir='./dataset/', transform=None):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.transform = transform
        
        self.img_paths = []
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                if len(row) == 0: continue
                image_path = os.path.join(img_dir, row[0])
                depth_path = os.path.join(img_dir, row[1])
                self.img_paths.append({ 'image_path': image_path, 'depth_path': depth_path })

    def get_image(self, file_name):
        image = Image.open(file_name)
        return image

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image = self.get_image(self.img_paths[index]['image_path'])
        depth = self.get_image(self.img_paths[index]['depth_path'])

        sample = {'image': image, 'depth': depth}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def get_train_test_dataloader(batch_size):
    train_dataset = MemoryDataset(csv_path='./dataset/nyu2_train.csv', img_dir='./dataset', transform=train_transform())
    test_dataset = MemoryDataset(csv_path='./dataset/nyu2_test.csv', img_dir='./dataset', transform=train_transform())

    return DataLoader(train_dataset, batch_size, shuffle=True), DataLoader(test_dataset, batch_size, shuffle=False)

