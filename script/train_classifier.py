# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from net import weights_init, Classifier


def main():
    parser = argparse.ArgumentParser(description='pytorch example: MNIST')
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--display', '-d', type=int, default=100,
                        help='Number of interval to show progress')
    args = parser.parse_args()

    batch_size = args.batch
    epoch_size = args.epoch
    display_interval = args.display

    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # transform to torch.Tensor
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_set = torchvision.datasets.CIFAR10(root='../cifar10_root', train=True, download=True, transform=train_trans)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_trans = transforms.Compose([
        transforms.ToTensor(),  # transform to torch.Tensor
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    test_set = torchvision.datasets.CIFAR10(root='../cifar10_root', train=False, download=True, transform=test_trans)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    net = Classifier()
    net.apply(weights_init)
    print(net)
    print()

    net.to(device)  # for GPU

    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)



    epoch_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(epoch_size):  # loop over the dataset multiple times

        running_loss = 0.0
        train_true = []
        train_pred = []
        # for itr, data in enumerate(train_loader, 0):
        for itr, data in enumerate(train_loader):
            # get the inputs
            images, labels = data

            train_true.extend(labels.tolist())

            images, labels = images.to(device), labels.to(device)   # for GPU

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_pred.extend(predicted.tolist())

            # print statistics
            running_loss += loss.item()
            if itr % display_interval == display_interval - 1:    # print every 100 mini-batches
                print('[epochs: {}, mini-batches: {}, images: {}] loss: {:.3f}'.format(
                    epoch + 1, itr + 1, (itr + 1) * batch_size, running_loss / display_interval))
                running_loss = 0.0

        test_true = []
        test_pred = []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                test_true.extend(labels.tolist())
                images, labels = images.to(device), labels.to(device)  # for GPU

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)            
                test_pred.extend(predicted.tolist())

        train_acc = accuracy_score(train_true, train_pred)
        test_acc = accuracy_score(test_true, test_pred)
        print('    epocs: {}, train acc.: {:.3f}, test acc.: {:.3f}'.format(epoch + 1, train_acc, test_acc))
        print()
        
        epoch_list.append(epoch + 1)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    print('Finished Training')

    print('Save Network')
    torch.save(net.state_dict(), 'model.pth')

    df = pd.DataFrame({'epoch': epoch_list,
                       'train/accuracy': train_acc_list,
                       'test/accuracy': test_acc_list})

    print('Save Training Log')
    df.to_csv('train_womix.log', index=False)


if __name__ == '__main__':
    main()
