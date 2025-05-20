import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


DATASET_PATH = './cats'


def get_dataloader(batch_size: int = 128, workers: int = 0):
    dataset = datasets.ImageFolder(root=DATASET_PATH,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    return dataloader
