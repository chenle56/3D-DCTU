import numpy as np
from skimage.measure import block_reduce
import skimage
import scipy.io
from ParamConfig import *

# slide_window = 64
# step_length = 32

stride=Patch_size
patch_size=Patch_size

def DataLoad_Test(test_size, test_data_dir, data_dim, in_channels, model_dim, data_dsp_blk, label_dsp_blk, start,
                  datafilename, dataname, truthfilename, truthname, scaler):
    label_set = []
    test_set = []
    for i in range(start, start + test_size):
        filename_seis = test_data_dir + 'georec_test/' + datafilename + str(i)
        print(filename_seis)
        data1_set = scipy.io.loadmat(filename_seis)
        h, w = data1_set[str(dataname)].shape[0], data1_set[str(dataname)].shape[1]
        data1_set = np.float32(data1_set[str(dataname)].reshape([h, w]))

        # normalize
        data1_set = scaler.fit_transform(data1_set)
        data1_set = data1_set.reshape([h, w, in_channels])

        for k in range(0, in_channels):
            temp = []
            data11_set = np.float32(data1_set[:, :, k])
            data11_set = np.float32(data11_set)
            # Data downsampling
            data_dsp_dim = data11_set.shape
            patches_test = get_patches(data11_set)
            for patch in patches_test:
                temp.append(patch)
            temp = np.expand_dims(temp, axis = 1)
            if k == 0:
                test_set1 = temp
            else:
                test_set1 = np.append(test_set1, temp, axis = 1)

        filename_label = test_data_dir + 'vmodel_test/' + truthfilename + str(i)
        data2_set = scipy.io.loadmat(filename_label)
        data2_set = np.float32(data2_set[str(truthname)].reshape(model_dim))

        # normalize
        data2_set = scaler.fit_transform(data2_set)

        label_dsp_dim = data2_set.shape
        patches_label = get_patches(data2_set)
        for patch in patches_label:
            label_set.append(patch)

        if i == start:
            test_set = test_set1
        else:
            test_set = np.append(test_set, test_set1, axis = 0)

    test_set = np.array(test_set, dtype = np.float32)
    label_set = np.array(label_set, dtype = np.float32)

    test_set = test_set.reshape((-1, in_channels, patch_size, patch_size))
    label_set = label_set.reshape((-1, 1, patch_size, patch_size))
    print('test_set=', test_set.shape, label_set.shape)
    return test_set, label_set, data_dsp_dim, label_dsp_dim


# downsampling function by taking the middle value
def decimate(a, axis):
    idx = np.round((np.array(a.shape)[np.array(axis).reshape(1, -1)] + 1.0) / 2.0 - 1).reshape(-1)
    downa = np.array(a)[:, :, idx[0].astype(int), idx[1].astype(int)]
    return downa

def get_patches(img):
    # get multiscale patches from a single image
    height, width = img.shape
    patches = []
    for i in range(0, height-patch_size+1, stride):
        for j in range(0, width-patch_size+1, stride):
            x = img[i:i+patch_size, j:j+patch_size]
            patches.append(x)
    return patches
