import argparse
import os
import sys
import time

import models
import torch
import utils


def get_args():
    parser = argparse.ArgumentParser(description='Testing script for MNIST')
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
    parser.add_argument('--batch_size', type=int, default=128, help='initial batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='for dataloader')
    parser.add_argument('--no_gpu', action='store_true', help='do not use gpu')

    args = parser.parse_args()
    return args


def main(args):
    path = os.path.join('saved_models', args.name)

    print('Starting testing')

    print(f'Command: {sys.argv}')
    for arg, value in sorted(vars(args).items()):
        print(f'Argument {arg}: {value}')

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    print(f'Using device: {device}')

    # load model
    model = models.get_model(args.model).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # dataloaders
    print('Loading dataloaders')
    testloader = utils.get_dataloader(
        args.directory, args.dataset, args.batch_size, args.num_workers, False
    )

    if os.path.exists(os.path.join(path, 'model.pt')):
        ckpt = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Loading pre-trained model from epoch {start_epoch}')
    else:
        sys.exit('Saved model not found')

    # test
    start_time = time.time()
    test_acc, test_loss = utils.test(model, testloader, criterion, device)
    test_time = time.time() - start_time
    print(f'Test  | Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}, Time: {test_time:.2f}s')


if __name__ == '__main__':
    args = get_args()
    main(args)
