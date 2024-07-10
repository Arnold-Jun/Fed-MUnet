import os
import pickle
import numpy as np
from collections import OrderedDict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from scipy.stats import dirichlet

class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, batch_size, patch_size, max_size):
        super().__init__(data, batch_size, None)
        self.oversample_foreground_percent = 1/3
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.max_size = max_size
    def generate_train_batch(self):
        # random select data
        bs = 0
        if self.batch_size > self.max_size:
            bs = self.max_size
        else:
            bs = self.batch_size
        sels = np.random.choice(list(self._data.keys()), bs, True)
        # read data, form slice
        images, labels = [], []
        for i, name in enumerate(sels):
            data = np.load(self._data[name]['path'], allow_pickle=True)
            if i < round(bs * (1 - self.oversample_foreground_percent)):
                force_fg = False
            else:
                force_fg = True
            if force_fg:
                locs = self._data[name]['locs']
                cls = np.random.choice(list(locs.keys()))
                indices = locs[cls][:, 0]
                sel_idx = np.random.choice(np.unique(indices))
                data = data[:, sel_idx]
                loc = locs[cls][indices == sel_idx]
                loc = loc[np.random.choice(len(loc))][1:]
                shape = np.array(data.shape[1:])
                center = shape // 2
                bias = loc - center
                pad_length = self.patch_size - shape
                pad_left = pad_length // 2 - bias
                pad_right = pad_length - pad_length // 2 + bias
                pad_left = np.clip(pad_left, 0, pad_length)
                pad_right = np.clip(pad_right, 0, pad_length)
                data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
            else:
                # randomly select slice
                sel_idx = np.random.choice(data.shape[1])
                data = data[:, sel_idx]
                shape = np.array(data.shape[1:])
                pad_length = self.patch_size - shape
                pad_left = pad_length // 2
                pad_right = pad_length - pad_length // 2
                data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
            images.append(data[:-1])
            labels.append(data[-1:])
        image = np.stack(images)
        label = np.stack(labels)
        return {'data': image, 'label': label}

def get_trainloader(config, num_clients):
    with open(os.path.join(config.SPLIT.ROOT, 'split_data.pkl'), 'rb') as f:
        splits = pickle.load(f)
    trains = splits['train']
    dataset = OrderedDict()
    alpha_param = 2

    weights = dirichlet.rvs(alpha_param * np.ones(num_clients))
    data_len = np.squeeze(len(trains) * weights)
    dataloader_list = []
    index = 0

    for i in data_len:
        train = trains[index: index + int(i)]
        index = index + int(i)
        for name in train:
            if name == "split_data":
                break
            dataset[name] = OrderedDict()
            dataset[name]['path'] = os.path.join(config.DATASET.ROOT, name+'.npy')
            with open(os.path.join(config.DATASET.ROOT, name+".pkl"), 'rb') as f:
                dataset[name]['locs'] = pickle.load(f)
        dataloader_list.append(DataLoader2D(dataset, config.TRAIN.BATCH_SIZE, config.TRAIN.PATCH_SIZE, int(i)))
    return dataloader_list


