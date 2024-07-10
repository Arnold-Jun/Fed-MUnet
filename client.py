import torch
from model.utils import norm
from model.utils.utils import AverageMeter
from torchvision.utils import save_image
from config.config import config
from model.seg.MUnet import MUnet
import torch.nn as nn


stages = config.MODEL.STAGES
midc = config.MODEL.MIDCHANNEL
model = MUnet(inc=4, outc=4, midc=midc, stages=stages)
model = nn.DataParallel(model, config.TRAIN.DEVICES)

class Client(object):

    def __init__(self, conf, train_dataset, criterion, scales, config, logger, c):
        self.conf = conf

        self.local_model = model

        self.train_dataset = train_dataset

        self.criterion = criterion
        self.scales = scales
        self.config = config
        self.logger = logger
        self.index = c

    def local_train(self, model, optim ,schedule, device):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optim_seg = optim
        sched_seg = schedule

        print_freq = 10
        lsegs = AverageMeter()
        self.local_model.train()

        for e in range(self.conf["local_epochs"]):
            self.logger.info('learning rate : {}'.format(optim_seg.param_groups[0]['lr']))
            num_iter = self.config.TRAIN.NUM_BATCHES
            for i in range(num_iter):
                data_dict = next(self.train_dataset)
                image = data_dict['data'].to(device)
                labels = data_dict['label']

                labels = [label.to(device) for label in labels]
                outs = self.local_model(image)

                lseg = self.criterion(outs, labels)
                loss = lseg
                lsegs.update(lseg.item(), self.config.TRAIN.BATCH_SIZE)

                optim_seg.zero_grad()
                loss.backward()
                optim_seg.step()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 12)

                if i % print_freq == 0:
                    msg = 'client {0} epoch: [{1}][{2}/{3}]\t' 'lseg {lseg.val:.3f} ({lseg.avg:.3f})\t'.format(
                         self.index, e, i, num_iter,
                         lseg=lsegs,
                     )
                    self.logger.info(msg)
                    bs = image.shape[0]
                    image = torch.cat(torch.split(image, 1, 1))
                    label = torch.cat(torch.split(labels[-1], 1, 1))
                    out = torch.argmax(torch.softmax(outs[-1], 1), dim=1, keepdim=True)
                    out = torch.cat(torch.split(out, 1, 1))
                    save_image(torch.cat([image, label, out], dim=0).data.to(device), f'tmp/train.png', nrow=bs,
                               scale_each=True, normalize=True)
                if self.conf["FlatClip"] :
                    model_norm = norm.model_norm(model, self.local_model)

                    norm_scale = min(1, self.conf['C'] / (model_norm))

                    for name, layer in self.local_model.named_parameters():
                        clipped_difference = norm_scale * (layer.data - model.state_dict()[name].cuda())
                        layer.data.copy_(model.state_dict()[name].cuda() + clipped_difference)
                else:
                    continue
            sched_seg.step()
            print("Client {0} Epoch {1} done." .format(self.index, e))
        if self.conf["FlatClip"]:
            diff = dict()
            for name, data in self.local_model.state_dict().items():
                diff[name] = (data - model.state_dict()[name]) * self.conf["lambda"]
            return {k: v.cpu().clone() for k, v in model.state_dict().items()}, diff
        else:
            return {k: v.cpu().clone() for k, v in self.local_model.state_dict().items()}

