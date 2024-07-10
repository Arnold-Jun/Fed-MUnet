import argparse, json
import random
import numpy as np
from server import *
from client import *
import torch
from config.config import config
import torch.backends.cudnn as cudnn
from model.utils.loss import DiceCELoss, MultiOutLoss
from model.utils.scheduler import PolyScheduler
from model.utils.utils import create_logger, setup_seed
from model.seg.MUnet import MUnet
from utils.dataloader import get_trainloader
from utils.augmenter import get_train_generator
import sys
sys.setrecursionlimit(10**7)


device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf', default="./config/conf.json")
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    stages = config.MODEL.STAGES

    midc = config.MODEL.MIDCHANNEL
    segnet = MUnet(inc=4, outc=4, midc=midc, stages=stages)
    segnet = nn.DataParallel(segnet, config.TRAIN.DEVICES).to(device)

    scales = [1 / 2 ** i for i in range(stages)][::-1]
    criterion = MultiOutLoss(DiceCELoss(), weights=scales)
    trainloader = get_trainloader(config, conf["no_models"])
    train_generator = get_train_generator(trainloader, scales, num_workers=config.NUM_WORKERS)

    logger = create_logger('./seg/log', 'train.log')

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    server = Server(conf, segnet, logger, device)
    clients = []

    for c in range(conf["no_models"]):
        clients.append(Client(conf, train_generator[c], criterion, scales, config, logger, c))

    for e in range(conf["global_epochs"]):
        candidates = random.sample(clients, conf["k"])

        weight_accumulator = {}
        client_params = []
        lens = len(candidates)

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
        diff_list = []

        for c in candidates:
            optim_seg = torch.optim.SGD(c.local_model.parameters(), lr=config.TRAIN.LR * np.exp(- config.TRAIN.LAMDA * e),
                                  weight_decay=config.TRAIN.WEIGHT_DECAY,
                                  momentum=0.95, nesterov=True)
            sched_seg = PolyScheduler(optim_seg, t_total=config.TRAIN.EPOCH)
            params, diff = c.local_train(server.global_model, optim_seg, sched_seg, device)
            client_params.append(params)
            diff_list.append(diff)
        for key in client_params[0].keys():
            weight_accumulator[key] = sum(client_params[i][key].to(device) + diff_list[i][key] for i in range(lens)) / lens

        server.model_aggregate(weight_accumulator)
        server.model_eval(device)
        print("Global Epoch {0} done.".format(e))
