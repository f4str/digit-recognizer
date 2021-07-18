import argparse
import logging
import os
import sys
import time

import models
import torch
import utils


def get_args():
    parser = argparse.ArgumentParser(description='Training script for MNIST')
    parser.add_argument('--name', type=str, required=True, help='name of the model')
    parser.add_argument('--directory', type=str, default='./data', help='path to dataset')
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help='dataset',
        choices=['mnist', 'fmnist', 'fashion-mnist', 'kmnist'],
    )
    parser.add_argument(
        '--model',
        type=str,
        default='convolutional',
        help='type of model to use',
        choices=['feedforward', 'convolutional', 'recurrent'],
    )
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='initial batch size')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='for dataloader')
    parser.add_argument('--patience', type=int, default=3, help='patience before early stopping')
    parser.add_argument('--no_save', action='store_true', help='do not save model')
    parser.add_argument('--no_gpu', action='store_true', help='do not use gpu')

    args = parser.parse_args()
    return args


def main(args):
    path = os.path.join('saved_models', args.name)

    # setup directories
    if not os.path.exists(path):
        os.makedirs(path)

    # setup logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(path, 'log.txt')))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    print(f'Logging to {os.path.join(path, "log.txt")}')

    logger.info('Starting training')
    logger.info(f'Command: {sys.argv}')
    for arg, value in sorted(vars(args).items()):
        logger.info(f'Argument {arg}: {value}')

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    logger.info(f'Using device: {device}')

    # load model
    model = models.get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # dataloaders
    print('Loading dataloaders')
    trainloader = utils.get_dataloader(
        args.directory, args.dataset, args.batch_size, args.num_workers, True
    )
    testloader = utils.get_dataloader(
        args.directory, args.dataset, args.batch_size, args.num_workers, False
    )

    start_epoch = 0
    best_acc = 0
    no_acc_change = 0

    if os.path.exists(os.path.join(path, 'model.pt')):
        ckpt = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        logger.info(f'Loading pre-trained model from epoch {start_epoch}')
    else:
        logger.info('Training from scratch')

    for epoch in range(start_epoch, args.epochs):
        logger.info(f'Epoch {epoch + 1}')

        # train
        start_time = time.time()
        train_acc, train_loss = utils.train(model, trainloader, optimizer, criterion, device)
        train_time = time.time() - start_time
        logger.info(
            f'Train | Accuracy: {train_acc:.2f}%, Loss: {train_loss:.4f}, Time: {train_time:.2f}s'
        )

        # test
        start_time = time.time()
        test_acc, test_loss = utils.test(model, testloader, criterion, device)
        test_time = time.time() - start_time
        logger.info(
            f'Test  | Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}, Time: {test_time:.2f}s'
        )

        if test_acc > best_acc:
            best_acc = test_acc
            no_acc_change = 0
            # save the model
            if not args.no_save:
                logger.info('Saving model')
                save_dict = {
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                }
                torch.save(save_dict, os.path.join(path, 'model.pt'))
        else:
            no_acc_change += 1

        if no_acc_change >= args.patience:
            logger.info('Patience exhausted, early stopping')
            break

    logger.info('\nFinished Training, final test using best model')

    # load best model
    if not args.no_save:
        ckpt = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(ckpt['state_dict'])
        logger.info(f'Loaded saved best model from epoch {ckpt["epoch"]}')
    else:
        logger.info('No saved model found, using last epoch trained')

    # test using best model
    start_time = time.time()
    test_acc, test_loss = utils.test(model, testloader, criterion, device)
    test_time = time.time() - start_time
    logger.info(f'Test  | Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}, Time: {test_time:.2f}s')


if __name__ == '__main__':
    args = get_args()
    main(args)
