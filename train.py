import argparse
import os.path

from model import *
from torchsummary import summary
import torch

from data import get_train_test_dataloader
from loss import depth_loss
from augmentation import normalize_depth
from util import AvgTracker

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train(batch_size, epochs, learning_rate, checkpoint_path):
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    train_loader, test_loader = get_train_test_dataloader(batch_size)

    for epoch in range(start_epoch, epochs):
        model.train()
        N = len(train_loader)

        loss_tracker = AvgTracker()

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            images = batch['image']
            depth_maps = batch['depth']
            normalized_depth_maps = normalize_depth(depth_maps)

            # todo: normalize and clip depth

            y_pred = model(images)

            loss = depth_loss(y_pred, normalized_depth_maps)
            loss_tracker.update(loss.data.item(), images.size(0))
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, i, N, loss=loss_tracker))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch-size', default=8, type=int, help='batch size')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='total number of epochs')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('-cp', '--checkpoint-path', default='checkpoint.tar', type=str, help='path where checkpoints are stored and loaded')
    args = parser.parse_args()

    train(args.batch_size, args.epochs, args.learning_rate, args.checkpoint_path)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
