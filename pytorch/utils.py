from tqdm import tqdm

import torch


def train(model, dataloader, optimizer, criterion, device):
    model.train()

    acc_total = 0
    loss_total = 0
    total = 0

    with tqdm(dataloader) as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            n = labels.size(0)
            optimizer.zero_grad()

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1).squeeze()
            acc = (preds == labels).float().sum()

            # backwards pass
            loss.backward()
            optimizer.step()

            total += n
            acc_total += acc.item()
            loss_total += loss.item() * n
            pbar.set_postfix(acc=100 * acc_total / total, loss=loss_total / total)
    
    acc = 100 * acc_total / total
    loss = loss_total / total

    return acc, loss


def test(model, dataloader, criterion, device):
    model.eval()

    acc_total = 0
    loss_total = 0
    total = 0

    with tqdm(dataloader) as pbar, torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            n = labels.size(0)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1).squeeze()
            acc = (preds == labels).float().sum()

            total += n
            acc_total += acc.item()
            loss_total += loss.item() * n
            pbar.set_postfix(acc=100 * acc_total / total, loss=loss_total / total)
    
    acc = 100 * acc_total / total
    loss = loss_total / total

    return acc, loss
