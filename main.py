import torch
from torch.utils.data import DataLoader

import numpy as np
import time
import sys
import os

from models import utils, caption
from datasets import coco
from configuration import Config
from engine import train_one_epoch, evaluate
import pdb
import torchvision.datasets as dset
from torchvision import io, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TRAIN_PCT = 0.95
NUM_WORKERS = 2
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-4
IMAGE_SIZE = (224, 224)

MAX_TEXT_LENGTH = 32

LABEL_MASK = -100


def main(config):
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model, criterion = caption.build_model(config)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)



    tfms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=0.5, 
                std=0.5
            )
    ]
    )
    descale = transforms.Compose(
        [
            transforms.Normalize(
                mean = [ 0., 0., 0. ],
                std = 1 / 0.5
            ),
            transforms.Normalize(
                mean = -0.5,
                std = [ 1., 1., 1. ]
            ),                           
        ]
    )

    target_tfm = lambda x: random.choice(x)

    dataset_train = dset.CocoCaptions("/home/ivsh/scratch/datasets/cococaption/train2017/", "/home/ivsh/scratch/datasets/cococaption/annotations/captions_train2017.json", tfms)
    dataset_val = dset.CocoCaptions("/home/ivsh/scratch/datasets/cococaption/val2017/", "/home/ivsh/scratch/datasets/cococaption/annotations/captions_val2017.json", tfms)
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint...")
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    print("Start Training..")
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, config.checkpoint)

        validation_loss = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")

        print()


if __name__ == "__main__":
    config = Config()
    main(config)
