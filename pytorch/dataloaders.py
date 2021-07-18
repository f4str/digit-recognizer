from torch.utils.data import DataLoader
import torchvision


def get_dataloader(directory: str, dataset: str, batch_size: int, num_workers: int, train: bool) -> DataLoader:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(root=directory, train=train, download=True, transform=transform)
    elif dataset in {'fmnist', 'fashion-mnist'}:
        dataset = torchvision.datasets.FashionMNIST(root=directory, train=train, download=True, transform=transform)
    elif dataset == 'kmnist':
        dataset = torchvision.datasets.KMNIST(root=directory, train=train, download=True, transform=transform)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return dataloader
