import os
import argparse
from train import main


def parse_args():
    parser = argparse.ArgumentParser(description='command for semi-segmentation model')
    parser.add_argument('--root', type=str, default='D:\\Data\\cifar', help='path of dataset')
    parser.add_argument('--num_classes', type=int, default=10, help='classes of the data')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--epochs', type=int, default=180, help='max epoch')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)
