import argparse

from model import *
from torchsummary import summary
import torch

from data import get_train_test_dataloader
from loss import depth_loss

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cpu')

def train(batch_size, epochs, learning_rate):
    model = Model().to(device)

    train_loader, test_loader = get_train_test_dataloader(batch_size)
    print(len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        N = len(train_loader)

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            images = batch['image'].to(device)
            depth_maps = batch['depth'].to(device)

            # todo: normalize and clip depth

            y_pred = model(images)

            loss = depth_loss(y_pred, depth_maps)
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, N))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch-size', default=4, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='total number of epochs')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    args = parser.parse_args()

    train(args.batch_size, args.epochs, args.learning_rate)

if __name__ == '__main__':
    main()
