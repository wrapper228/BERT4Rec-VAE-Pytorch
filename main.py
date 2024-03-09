import torch

from options import args
from models.bert import BERTModel
from dataloaders import make_dataloaders
from trainers import trainer_factory
from utils import *


def train():
    train_loader, val_loader, test_loader = make_dataloaders(args)

    model = BERTModel(args)

    some_batch = next(iter(train_loader))
    seq, _ = some_batch
    print(seq.shape)
    print(seq[0, ...].unsqueeze(0).shape)
    print(model(seq[0, ...].unsqueeze(0)).shape)

    export_root = setup_train(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    # цель: вытащить этот процесс трейна наружу. первым делом понять что выплевывает model(batch)

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
