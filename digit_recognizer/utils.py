import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm


def get_dataloader(
    directory: str, dataset: str, batch_size: int, num_workers: int, train: bool
) -> DataLoader:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if dataset == 'mnist':
        dset = torchvision.datasets.MNIST(
            root=directory, train=train, download=True, transform=transform
        )
    elif dataset == 'qmnist':
        dset = torchvision.datasets.QMNIST(
            root=directory, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    return dataloader


def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total = 0
    acc_total = 0
    loss_total = 0

    with tqdm(dataloader) as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            n = labels.size(0)
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().sum()

            # backwards pass
            loss.backward()
            optimizer.step()

            # update metrics
            total += n
            acc_total += acc.item()
            loss_total += loss.item() * n
            pbar.set_postfix(acc=100 * acc_total / total, loss=loss_total / total)

    acc = 100 * acc_total / total
    loss = loss_total / total

    return acc, loss


def test(model, dataloader, criterion, device):
    model.eval()

    total = 0
    acc_total = 0
    loss_total = 0

    with tqdm(dataloader) as pbar, torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            n = labels.size(0)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().sum()

            # update metrics
            total += n
            acc_total += acc.item()
            loss_total += loss.item() * n
            pbar.set_postfix(acc=100 * acc_total / total, loss=loss_total / total)

    acc = 100 * acc_total / total
    loss = loss_total / total

    return acc, loss
