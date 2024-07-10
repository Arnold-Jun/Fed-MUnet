import torch
import os, pickle
import numpy as np
import torch.nn as nn
import statistics as stat
from config.config import config
from model.seg.MUnet import MUnet
from medpy.metric.binary import *
from model.utils.function import inference
import torch.backends.cudnn as cudnn
from model.utils.utils import create_logger, setup_seed

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
@torch.no_grad()
def inference(model, logger, config, dataset, metrics):
    model.eval().to(device)
    perfs = {}

    for metric in metrics:
        perfs[metric.__name__] = {'WT': [], 'ET': [], 'TC': []}
    nonline = nn.Softmax(dim=1)

    with open(os.path.join(config.SPLIT_eva.ROOT, 'split_data.pkl'), 'rb') as f:
        splits = pickle.load(f)

    valids = splits[dataset]
    for name in valids:
        data = np.load(os.path.join(config.DATASET_eva.ROOT, name+'.npy'))

        shape = np.array(data.shape[2:])
        pad_length = config.TRAIN.PATCH_SIZE - shape
        pad_left = pad_length // 2
        pad_right = pad_length - pad_length // 2
        pad_left = np.clip(pad_left, 0, pad_length)
        pad_right = np.clip(pad_right, 0, pad_length)
        data = np.pad(data, ((0, 0), (0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))

        image = torch.from_numpy(data[:-1]).permute(1, 0, 2, 3).to(device)
        label = data[-1]
        out_list = [model(torch.tensor(np.expand_dims(image[i].cpu(), 0)).to(device)) for i in range(image.shape[0])]
        out_list = torch.cat(out_list, dim=0)

        out = nonline(out_list)
        pred = torch.argmax(out, dim=1).cpu().numpy()

        mask = image.permute(1, 0, 2, 3)[0].cpu().numpy() != 0

        for metric in metrics:
            predi = pred if metric.__name__ == 'hd95' else pred[mask]
            labeli = label if metric.__name__ == 'hd95' else label[mask]
            perfs[metric.__name__]['WT'].append(metric(predi > 0, labeli > 0))

            if 3 in label:
                y_true_class3 = (labeli == 3).astype(int)
                y_pred_class3 = (predi == 3).astype(int)
                perfs[metric.__name__]['ET'].append(metric((predi == 3).astype(int), (labeli == 3).astype(int)))
            if 2 in label:
                perfs[metric.__name__]['TC'].append(metric(predi >= 2, labeli >= 2))
    for metric in perfs.keys():
        et = perfs[metric]['ET']
        tc = perfs[metric]['TC']
        wt = perfs[metric]['WT']
        logger.info(f'------------ {metric} ------------')
        print(f'ET mean / std: {stat.mean(et)} / {stat.stdev(et)}')
        logger.info(f'ET mean / std: {stat.mean(et)} / {stat.stdev(et)}')
        logger.info(f'TC mean / std: {stat.mean(tc)} / {stat.stdev(tc)}')
        logger.info(f'WT mean / std: {stat.mean(wt)} / {stat.stdev(wt)}')

def main():
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = MUnet(inc=4, outc=4, midc=16, stages=config.MODEL.STAGES)
    model = nn.DataParallel(model, config.TRAIN.DEVICES)
    model.load_state_dict(torch.load('./experiments/model_best.pth', map_location=torch.device('cpu')))

    logger = create_logger('log', 'test.log')
    inference(model, logger, config, dataset='test', metrics=[dc, jc, hd95, sensitivity, precision, specificity])

if __name__ == '__main__':
    main()
