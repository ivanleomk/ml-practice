from torchvision import datasets, transforms
import torch
import os

DOWNLOAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TRAIN_PATH = os.path.join(DOWNLOAD_PATH, "train")
TEST_PATH = os.path.join(DOWNLOAD_PATH, "test")

os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)


def read_mnist_dataset():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    trainset = datasets.MNIST(
        TRAIN_PATH, download=True, train=True, transform=transform
    )
    valset = datasets.MNIST(TEST_PATH, download=True, train=False, transform=transform)
    return trainset, valset


if __name__ == "__main__":
    trainset, valset = read_mnist_dataset()
