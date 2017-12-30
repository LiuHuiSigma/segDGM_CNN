import nibabel as nib
import numpy as np
import itertools
from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
from matplotlib import pyplot as plt
import bcolz
import random

precision_global = 'float32'

# General utils for loading and saving data
def read_data(case_idx, type_name, loc='datasets'):
    file_name = '{0}/{1}/{2}.nii.gz'.format(loc, type_name, case_idx)
    return nib.load(file_name).get_data()

def save_data(data, case_idx, type_name, loc='results'):
    file_name_ex = '{0}/{1}/{2}.nii.gz'.format('datasets', 'QSM', case_idx)
    file_name = '{0}/{1}/{2}.nii.gz'.format(loc, type_name, case_idx)
    nib.save(nib.Nifti1Image(data.astype('uint8'), None, nib.load(file_name_ex).header), file_name)

def save_array(file_name, arr):
    c = bcolz.carray(arr, rootdir=file_name, mode='w')
    c.flush()
    
def load_array(file_name):
    return bcolz.open(file_name)[:]


# Data preparation utils
def extract_patches(volume, patch_shape, extraction_step) :
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)
    
    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)

def build_set(input_vols, label_vols, extraction_step=(9, 9, 9), patch_shape=(27, 27, 27), predictor_shape=(9, 9, 9), mask=None) :
    #patch_shape = (27, 27, 27)
    #label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]
    label_selector = [slice(None)] + [slice(int((patch_shape[i]-predictor_shape[i])/2), int((patch_shape[i]-predictor_shape[i])/2+predictor_shape[i])) for i in range(3)]
    
    num_classes = len(np.unique(label_vols))
    num_channel = input_vols.shape[1]

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, num_channel) + patch_shape, dtype=precision_global)
    y = np.zeros((0, predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)
    for idx in range(len(input_vols)) :
        print(idx)
        y_length = len(y)

        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step)
        label_patches = label_patches[label_selector]

        # Select only those who are important for processing
        valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)
        
        # Enforce including patches covering mask
        if type(mask) is not type(None):
            mask_patches = extract_patches(mask[idx], patch_shape, extraction_step)
            mask_patches = mask_patches[label_selector]
            valid_idxs = np.where((np.sum(label_patches, axis=(1, 2, 3)) != 0) | (np.sum(mask_patches, axis=(1, 2, 3)) != 0))
        
        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        # Extend volume
        x = np.vstack((x, np.zeros((len(label_patches), num_channel) + patch_shape, dtype=precision_global)))
        y = np.vstack((y, np.zeros((len(label_patches), predictor_shape[0]*predictor_shape[1]*predictor_shape[2], num_classes), dtype=precision_global)))
        
        for i in range(len(label_patches)) :
            y[i+y_length, :, :] = np_utils.to_categorical(label_patches[i, : ,: ,:], num_classes).reshape((-1, num_classes))

        del label_patches

        # Sampling strategy: reject samples which labels are only zero
        for i_channel in range(num_channel):
            input_patches = extract_patches(input_vols[idx, i_channel], patch_shape, extraction_step)
            x[y_length:, i_channel, :, :, :] = input_patches[valid_idxs]
        del input_patches

    return x, y


# Random shuffle (1st dim)
def shuffle(data, idxs = None):
    N = len(data)
    
    if idxs == None:
        idxs = list(range(N))
        for i in range(N-1, -1, -1):
            j = random.randint(0, i)
            idxs[i], idxs[j] = idxs[j], idxs[i]
    
    idxs_target = [0] * N
    for i, idx in enumerate(idxs):
        idxs_target[idx] = i
    
    for i in range(N):
        while i != idxs_target[i]:
            j = idxs_target[i]
            data[i], data[j] = data[j], data[i]
            idxs_target[i], idxs_target[j] = idxs_target[j], idxs_target[i]
        
    return idxs
        
        
    
    
    
    

# Reconstruction utils
def generate_indexes(patch_shape, expected_shape) :
    ndims = len(patch_shape)

    #poss_shape = [patch_shape[i+1] * (expected_shape[i] // patch_shape[i+1]) for i in range(ndims-1)]
    poss_shape = [patch_shape[i+1] * ((expected_shape[i]-18) // patch_shape[i+1]) + 18 for i in range(ndims-1)]

    #idxs = [range(patch_shape[i+1], poss_shape[i] - patch_shape[i+1], patch_shape[i+1]) for i in range(ndims-1)]
    idxs = [range(9, poss_shape[i] - 9, patch_shape[i+1]) for i in range(ndims-1)]

    return itertools.product(*idxs)

def reconstruct_volume(patches, expected_shape) :
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape, dtype=precision_global)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img

# Utils for plotting
def plots(ims, figsize=(12,6), rows=1, scale=None, interp=False, titles=None):
    
    if scale != None:
        lo, hi = scale
        ims = (ims - lo)/(hi - lo) * 255
        
    if(ims.ndim == 2):
        ims = np.tile(ims, (1,1,1));
    
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1] != 3):
            ims = np.tile(ims[:,:,:,np.newaxis], (1,1,1,3));
            
    #print(ims.shape)
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')