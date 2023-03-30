from model import *
from torchsummary import summary
import torch

from data import get_train_test_dataloader

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cpu')

model = Model().to(device)

train_loader, test_loader = get_train_test_dataloader(8)


import matplotlib.pyplot as plt

X, Y = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
