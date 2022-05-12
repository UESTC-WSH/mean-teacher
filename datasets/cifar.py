import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def dataloader(args):
    train_dataset = datasets.CIFAR10(root=args.root, train=True, transform=transform_train, download=False)
    test_dataset = datasets.CIFAR10(root=args.root, train=False, transform=transform_test, download=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description='command for semi-segmentation model')
    parser.add_argument('--root', type=str, default='D:\\Data\\cifar', help='path of dataset')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_loader, test_loader = dataloader(args)
    print(len(train_loader.dataset), len(test_loader.dataset))
    train_loader = iter(train_loader)
    x_train, y_train = next(train_loader)
    print(x_train.shape)
    print(len(train_loader))
    '''
    for x_train, y_train in train_loader:
        print(x_train.shape)
        print(y_train.shape)
        to_pil_image = transforms.ToPILImage()
        img = to_pil_image(x_train[0])
        img.show()
        break
    '''
