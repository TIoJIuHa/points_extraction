import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from utils.utils import load_checkpoint


def get_save_path(output_dir):
    """
    This func
    :param output_dir:
    :return:
    """
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save everything to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path

def worker_init_fn(worker_id):
   """The function is designed for pytorch multi-process dataloader."""
   base_seed = torch.IntTensor(1).random_().item()
   np.random.seed(base_seed + worker_id)


def dataLoader(config, dataset='syn', warp_input=False, export_task='train'):
    training_params = config.get('training', {})
    workers_test = training_params.get('workers_test', 0) # 16
    logging.info(f"workers_test: {workers_test}")
    logging.info(f"load dataset from : {dataset}")
    Dataset = get_module('utils', dataset)
    test_set = Dataset(
        export=True,
        task=export_task,
        **config['data'],
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False,
        pin_memory=True,
        num_workers=workers_test,
        worker_init_fn=worker_init_fn
    )
    return {'test_set': test_set, 'test_loader': test_loader}


def get_module(path, name):
    import importlib
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)


def get_model(name):
    mod = __import__('models.{}'.format(name), fromlist=[''])
    return getattr(mod, name)


def modelLoader(model='SuperPointNet', **options):
    logging.info("=> creating model: %s", model)
    net = get_model(model)
    net = net(**options)
    return net


def pretrainedLoader(net, optimizer, epoch, path, mode='full', full_path=False):
    if full_path == True:
        checkpoint = torch.load(path)
    else:
        checkpoint = load_checkpoint(path)

    if mode == 'full':
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['n_iter']
    else:
        net.load_state_dict(checkpoint)
    return net, optimizer, epoch
