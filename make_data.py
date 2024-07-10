import random
import os, pickle
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from scipy.ndimage import binary_fill_holes

def get_bbox(inp):
    coords = np.where(inp != 0)
    minz = np.min(coords[0])
    maxz = np.max(coords[0]) + 1
    minx = np.min(coords[1])
    maxx = np.max(coords[1]) + 1
    miny = np.min(coords[2])
    maxy = np.max(coords[2]) + 1
    return slice(minz, maxz), slice(minx, maxx), slice(miny, maxy)

def convert_seg(seg):
    """ convert brats labels from {0, 1, 2, 4} to {0, 1, 2, 3} """
    new_seg = np.zeros_like(seg)
    new_seg[seg == 4] = 3
    new_seg[seg == 2] = 1
    new_seg[seg == 1] = 2
    return new_seg

def convert(data_path, out_path):

    names = os.listdir(data_path)
    for name in names:
        flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}_flair.nii.gz')))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}_t1.nii.gz')))
        t1ce = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}_t1ce.nii.gz')))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}_t2.nii.gz')))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_path, name, f'{name}_seg.nii.gz')))
        img = np.stack([flair, t1, t1ce, t2]).astype(np.float32)
        seg = convert_seg(seg)
        # crop foreground regions
        mask = np.zeros_like(seg).astype(bool)
        for i in range(len(img)):
            mask = mask | (img[i] != 0)
        mask = binary_fill_holes(mask)
        bbox = get_bbox(mask)
        img = img[:, bbox[0], bbox[1], bbox[2]]
        seg = seg[bbox[0], bbox[1], bbox[2]]
        mask = mask[bbox[0], bbox[1], bbox[2]]
        # normalization
        for i in range(len(img)):
            img[i][mask] = (img[i][mask] - img[i][mask].min()) / (img[i][mask].max() - img[i][mask].min())
            img[i][mask == 0] = 0
        # compensate label imbalance
        approx_nsamp = 10000
        samp_locs = OrderedDict()
        for cls in [1, 2, 3]:
            locs = np.argwhere(seg == cls)
            nsamp = min(approx_nsamp, len(locs))
            nsamp = max(nsamp, int(np.ceil(0.1 * len(locs))))
            samp = locs[random.sample(range(len(locs)), nsamp)]
            if len(samp) != 0:
                samp_locs[cls] = samp
        data = np.concatenate([img, seg[None]])
        np.save(os.path.join(out_path, f'{name}.npy'), data)
        with open(os.path.join(out_path, f'{name}.pkl'), 'wb') as f:
            pickle.dump(samp_locs, f)

def data_split(directory):

    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    pkl_files = [os.path.splitext(f)[0] for f in pkl_files]

    random.shuffle(pkl_files)

    total_files = len(pkl_files)
    train_size = int(total_files * 0.7)
    val_size = int(total_files * 0.2)

    train_files = pkl_files[:train_size]
    val_files = pkl_files[train_size:train_size + val_size]
    test_files = pkl_files[train_size + val_size:]

    data_split = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    output_file1 = './dataset/data/split_data.pkl'
    output_file2 = './dataset/data/split_data.npy'

    np.save(output_file2, data_split)
    with open(output_file1, 'wb') as f:
        pickle.dump(data_split, f)

    print(f"Data split saved to {output_file1}")

data_path = './data/raw'
directory = './dataset/proceed'

convert(data_path, directory)
data_split(directory)