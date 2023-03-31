from model import *
from torchsummary import summary
import torch

from data import get_train_test_dataloader

BATCH_SIZE = 1

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cpu')

model = Model().to(device)

train_loader, test_loader = get_train_test_dataloader(1)


import matplotlib.pyplot as plt

batch = next(iter(train_loader))
X = batch['image']
Y = batch['depth']
print(f"X batch shape: {X.size()}")
print(f"Y batch shape: {Y.size()}")

# img = X[0]
# label = Y[0]
# plt.imshow(img.permute((1, 2, 0)), cmap="gray")
# plt.show()

output = model(X)
print(output.shape)

plt.imshow(output[0].detach().permute((1, 2, 0)))
plt.show()
