import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
# from dataloaders import make_dataloaders
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    # train_loader, val_loader, test_loader = make_dataloaders(file_path)

    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
