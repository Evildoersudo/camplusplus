# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.epoch import EpochCounter, EpochLogger


parser = argparse.ArgumentParser(description='Speaker Network Training')
parser.add_argument('--config', default='', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')
parser.add_argument(
    '--init_model',
    default='',
    type=str,
    help='Optional pretrained embedding model checkpoint used to initialize embedding_model before training.',
)

def main():
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)

    # Fall back to a normal single-process training loop when distributed
    # environment variables are absent. This is the practical path on Windows.
    distributed = 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if distributed:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        world_size = 1

    # Prefer CUDA whenever it is available. The --gpu argument only controls
    # which device index to select.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if args.gpu is not None and len(args.gpu) > 0:
            gpu = int(args.gpu[rank % len(args.gpu)])
        else:
            gpu = 0
        torch.cuda.set_device(gpu)
    else:
        gpu = None

    if distributed:
        backend = 'nccl' if use_cuda else 'gloo'
        dist.init_process_group(backend=backend)

    set_seed(args.seed)

    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger('%s/train.log' % config.exp_dir)
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if use_cuda:
        logger.info(f"Use GPU: {gpu} for training.")
    else:
        logger.info("Use CPU for training.")

    # dataset
    train_dataset = build('dataset', config)
    # dataloader
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        config.dataloader['args']['sampler'] = train_sampler
        config.dataloader['args']['batch_size'] = int(config.batch_size / world_size)
    else:
        train_sampler = None
        config.dataloader['args'].pop('sampler', None)
        config.dataloader['args']['shuffle'] = True
        config.dataloader['args']['batch_size'] = int(config.batch_size)
    train_dataloader = build('dataloader', config)

    # model
    embedding_model = build('embedding_model', config)
    if args.init_model:
        # Fine-tuning reuses only the embedding network weights. The classifier
        # is always rebuilt for the current training speaker set.
        pretrained_state = torch.load(args.init_model, map_location='cpu')
        load_msg = embedding_model.load_state_dict(pretrained_state, strict=False)
        logger.info(f"Initialized embedding_model from {args.init_model}")
        logger.info(f"Missing keys when loading init_model: {load_msg.missing_keys}")
        logger.info(f"Unexpected keys when loading init_model: {load_msg.unexpected_keys}")
    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * 3
    else:
        config.num_classes = len(config.label_encoder)

    classifier = build('classifier', config)
    model = nn.Sequential(embedding_model, classifier)
    device = torch.device('cuda', gpu) if use_cuda else torch.device('cpu')
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu] if use_cuda else None)

    # optimizer
    config.optimizer['args']['params'] = model.parameters()
    optimizer = build('optimizer', config)

    # loss function
    criterion = build('loss', config)

    # scheduler
    config.lr_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    lr_scheduler = build('lr_scheduler', config)
    config.margin_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    margin_scheduler = build('margin_scheduler', config)

    # others
    epoch_counter = build('epoch_counter', config)
    checkpointer = build('checkpointer', config)

    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))

    # resume from a checkpoint
    if args.resume:
        checkpointer.recover_if_possible(device=device)

    cudnn.benchmark = True

    for epoch in epoch_counter:
        if distributed:
            train_sampler.set_epoch(epoch)

        # train one epoch
        train_stats = train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            lr_scheduler,
            margin_scheduler,
            logger,
            config,
            rank,
            device,
        )

        if rank == 0:
            # log
            epoch_logger.log_stats(
                stats_meta={"epoch": epoch},
                stats=train_stats,
            )
            # save checkpoint
            if epoch % config.save_epoch_freq == 0:
                checkpointer.save_checkpoint(epoch=epoch)

        if distributed:
            dist.barrier()

def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, margin_scheduler, logger, config, rank, device):
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Data', ':6.3f')
    train_stats.add('Loss', ':.4e')
    train_stats.add('Acc@1', ':6.2f')
    train_stats.add('Lr', ':.3e')
    train_stats.add('Margin', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        train_stats,
        prefix="Epoch: [{}]".format(epoch)
    )

    #train mode
    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        # data loading time
        train_stats.update('Data', time.time() - end)

        # update
        iter_num = (epoch-1)*len(train_loader) + i
        lr_scheduler.step(iter_num)
        margin_scheduler.step(iter_num)

        x = x.to(device, non_blocking=(device.type == 'cuda'))
        y = y.to(device, non_blocking=(device.type == 'cuda'))

        # compute output
        output = model(x)
        loss = criterion(output, y)
        acc1 = accuracy(output, y)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # recording
        train_stats.update('Loss', loss.item(), x.size(0))
        train_stats.update('Acc@1', acc1.item(), x.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        train_stats.update('Margin', margin_scheduler.get_margin())
        train_stats.update('Time', time.time() - end)

        if rank == 0 and i % config.log_batch_freq == 0:
            logger.info(progress.display(i))

        end = time.time()

    key_stats={
        'Avg_loss': train_stats.avg('Loss'),
        'Avg_acc': train_stats.avg('Acc@1'),
        'Lr_value': train_stats.val('Lr')
    }
    return key_stats

if __name__ == '__main__':
    main()
