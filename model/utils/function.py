import torch
import os, pickle
import numpy as np
import torch.nn as nn
from medpy.metric.binary import dc
from model.utils.utils import AverageMeter
import random

@torch.no_grad()
def inference(model, logger, config, dataset, device):
    print("-----------------inference------------------")
    model.eval().to(device)
    perfs = {'WT': AverageMeter(), 'ET': AverageMeter(), 'TC': AverageMeter()}
    nonline = nn.Softmax(dim=1)
    with open(os.path.join(config.SPLIT.ROOT, 'split_data.pkl'), 'rb') as f:
        splits = pickle.load(f)

    valids = random.sample(splits[dataset], 50)

    for name in valids:
        data = np.load(os.path.join(config.DATASET.ROOT, name+'.npy'))
        # pad slice
        shape = np.array(data.shape[2:])
        pad_length = config.TRAIN.PATCH_SIZE - shape
        pad_left = pad_length // 2
        pad_right = pad_length - pad_length // 2
        pad_left = np.clip(pad_left, 0, pad_length)
        pad_right = np.clip(pad_right, 0, pad_length)
        data = np.pad(data, ((0, 0), (0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
        # run inference
        image = torch.from_numpy(data[:-1]).permute(1, 0, 2, 3).to(device)
        label = data[-1]
        out_list = [model(torch.tensor(np.expand_dims(image[i].cpu(), 0)).to(device)) for i in range(image.shape[0])]
        out_list = torch.cat(out_list, dim=0)
        out_list = nonline(out_list)
        pred = torch.argmax(out_list, dim=1).cpu().numpy()
        # quantitative analysis
        perfs['WT'].update(dc(pred > 0, label > 0))
        if 3 in label:
            perfs['ET'].update(dc(pred == 3, label == 3))
        if 2 in label:
            perfs['TC'].update(dc(pred >= 2, label >= 2))
    for c in perfs.keys():
        logger.info(f'class {c} dice mean: {perfs[c].avg}')
    logger.info('------------ ----------- ------------')
    perf = np.mean([perfs[c].avg for c in perfs.keys()])
    return perf

