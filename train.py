import argparse
import os.path

from model import *
from torchsummary import summary
import torch

from data import get_train_test_dataloader
from loss import depth_loss
from augmentation import normalize_depth

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(batch_size, epochs, learning_rate, checkpoint_path):
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    train_loader, test_loader = get_train_test_dataloader(batch_size)

    for epoch in range(start_epoch, epochs):
        model.train()
        N = len(train_loader)

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            images = batch['image'].to(device)
            depth_maps = batch['depth'].to(device)
            normalized_depth_maps = normalize_depth(depth_maps)

            # todo: normalize and clip depth

            y_pred = model(images)

            loss = depth_loss(y_pred, normalized_depth_maps)
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, N))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch-size', default=4, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='total number of epochs')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('-cp', '--checkpoint-path', default='', type=str, help='path where checkpoints are stored and loaded')
    args = parser.parse_args()

    train(args.batch_size, args.epochs, args.learning_rate, args.checkpoint_path)

if __name__ == '__main__':
    main()
