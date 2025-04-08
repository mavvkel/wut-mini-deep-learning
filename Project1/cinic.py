import torch
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def get_loaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    cinic_directory = './dataset'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/train',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
        batch_size=batch_size, shuffle=True)

    valloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/valid',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
        batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/test',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
        batch_size=batch_size, shuffle=True)

    return trainloader, valloader, testloader


def get_augment_loaders(batch_size: int, transform: transforms.Compose) -> tuple[DataLoader, DataLoader, DataLoader]:
    cinic_directory = './dataset'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            cinic_directory + '/train', transform=transform
        ),
        batch_size=batch_size,
        shuffle=True
    )

    valloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/valid',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
        batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/test',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
        batch_size=batch_size, shuffle=True)

    return trainloader, valloader, testloader


def get_augment_transforms():
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    # Standardowe augmentacje (Testujemy każdą osobno)
    transform_flip = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])

    transform_rotation = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])

    transform_jitter = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])

    # Wszystkie 3 standardowe augmentacje
    transform_standard = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])

    # Zaawansowana augmentacja (AutoAugment)
    transform_autoaugment = transforms.Compose([
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])

    transform_auto_standard = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean, std=cinic_std),
    ])

    return {
        "flip": transform_flip,
        "rotation": transform_rotation,
        "jitter": transform_jitter,
        "standard": transform_standard,
        "autoaugment": transform_autoaugment,
        "auto_standard": transform_auto_standard
    }


if __name__ == '__main__':
    def imshow(inp, title=None):
        """Display image for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.figure(figsize=(10, 22))
        plt.imshow(inp)
        plt.ylabel("Classes", fontsize=14)
        plt.yticks(
            np.arange(18, 358, 34),
            [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ],
            fontsize=14,
        )
        plt.xlabel("Instances", fontsize=14)
        plt.xticks(
            np.arange(18, 188, 34),
            [f"{i}" for i in range(1, 6)],
            fontsize=14,
        )
        plt.xticks()
        if title is not None:
            plt.title(title, fontsize=20)
        plt.pause(0.001)  # pause a bit so that plots are updated

    trainloader, _, _ = get_loaders(256)
    # Get a batch of training data
    inputs, classes = next(iter(trainloader))

    inputs_arr = inputs.numpy()
    classes_arr: np.ndarray = classes.numpy()
    sorted = np.zeros((50, 3, 32, 32))

    for cl in range(10):
        instances = inputs_arr[classes_arr == cl]
        instances = instances[0:5]
        sorted[(cl*5):(cl*5+5)] = instances.copy()

    # Make a grid from batch
    out = torchvision.utils.make_grid(torch.from_numpy(sorted), nrow=5)

    imshow(out, "Sample CINIC-10 images for every class")
    print('y')
