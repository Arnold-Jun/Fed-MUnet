import numpy as np
from skimage.transform import resize
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

class DownsampleSegTransform(AbstractTransform):
    """
    transform segmentation label to a list of labels scaled according to deep supervision scales using nearestNeighbor interpolation
    """
    def __init__(self, scales=(1., 0.5, 0.25), label_key='label'):
        super().__init__()
        self.scales = scales
        self.label_key = label_key

    def __call__(self, **data_dict):
        label = data_dict[self.label_key]
        axes = list(range(2, len(label.shape)))
        labels = []
        for s in self.scales:
            if s == 1.:
                labels.append(label)
            else:
                new_shape = np.array(label.shape).astype(float)
                for a in axes:
                    new_shape[a] *= s
                new_shape = np.round(new_shape).astype(int)
                out_label = np.zeros(new_shape, dtype=label.dtype)
                for i in range(label.shape[0]):
                    for j in range(label.shape[1]):
                        out_label[i, j] = resize(label[i, j].astype(float), new_shape[2:], order=0, mode='edge', clip=True, anti_aliasing=False).astype(label.dtype)
                labels.append(out_label)
        data_dict[self.label_key] = labels
        return data_dict
    
def get_train_generator(trainloader, scales, num_workers):
    transforms = []
    transforms.extend([DownsampleSegTransform(scales=scales, label_key='label')])
    transforms.extend([NumpyToTensor(keys=['data', 'label'], cast_to='float')])

    transforms = Compose(transforms)
    batch_generator_list = []

    for i in trainloader:
        batch_generator = MultiThreadedAugmenter(
        data_loader=i,
        transform=transforms,
        num_processes=num_workers,
        pin_memory=True
    )
        batch_generator_list.append(batch_generator)
    return batch_generator_list

