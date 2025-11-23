import numpy as np
import torch
from skimage.measure import block_reduce

import skimage
import scipy.io
from IPython.core.debugger import set_trace
from skimage import transform
from ParamConfig import *

stride = 64
patch_size = Patch_size
# scales = [1,0.8]
scales = [1]
aug_times = 8


def DataLoad_Train(train_size, train_data_dir, data_dim, in_channels, model_dim, data_dsp_blk, label_dsp_blk, start,
                   datafilename, dataname, truthfilename, truthname, batchSize):
    train_set = []
    label_set = []
    for i in range(start, start + train_size):
        filename_seis = train_data_dir + 'georec_train/' + datafilename + str(i)
        print(filename_seis)
        # Load .mat data
        data1_set = scipy.io.loadmat(filename_seis)
        print(data1_set.keys())
        data1_set = np.float32(data1_set[str(dataname)].transpose(2, 0, 1)) #[h,w,d] -->[d,h,w]
        data1_set = np.expand_dims(data1_set, axis=3)
        data_dsp_dim = data1_set.shape # [d,h,w,c], c=1
        patches = get_patches(data1_set,scaler)
        for patch in patches:
            train_set.append(patch)
        filename_label = train_data_dir + 'vmodel_train/' + truthfilename + str(i)
        data2_set = scipy.io.loadmat(filename_label)
        data2_set = np.float32(data2_set[str(truthname)].transpose(2, 0, 1))
        data2_set = np.expand_dims(data2_set, axis=3)
        label_dsp_dim = data2_set.shape # [d,h,w,c], c=1
        patches = get_patches(data2_set, scaler)
        for patch in patches:
            label_set.append(patch)
    train_set = np.stack(train_set)
    label_set = np.stack(label_set)
    # dimention [b, c, d, h, w]
    train_set = train_set.transpose(0, 4, 1, 2, 3)
    label_set = label_set.transpose(0, 4, 1, 2, 3)

    print('train_set=', train_set.shape, label_set.shape)
    return train_set, label_set, data_dsp_dim, label_dsp_dim


# downsampling function by taking the middle value
def decimate(a, axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1, -1)] + 1.0) / 2.0 - 1).reshape(-1)
    downa = np.array(a)[:, :, idx[0].astype(int), idx[1].astype(int)]
    return downa


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def get_patches(img, scaler):
    # get multiscale patches from a single image
    d, h, w, _ = img.shape
    patches = []
    for s in scales:
        d_scaled, h_scaled, w_scaled = int(d * s), int(h * s), int(w * s)
        for d in range(0, d_scaled - patch_size + 1, stride):
            for i in range(0, h_scaled - patch_size + 1, stride):
                for j in range(0, w_scaled - patch_size + 1, stride):
                    data = img[d:d + patch_size, i:i + patch_size, j:j + patch_size, :]
                    patches.append(data)
    return patches
