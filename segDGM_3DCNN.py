
# coding: utf-8

# # Problem configuration



# In[102]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import nibabel as nib
import numpy as np
from scipy import ndimage
from keras import backend as K
from sklearn.metrics import accuracy_score, precision_score

from utils import *
from model_FCNN import generate_model


# In[103]:


from importlib import reload

import keras
reload(keras)
from keras import backend as K

import utils
reload(utils)
from utils import *

import model_FCNN
reload(model_FCNN)
from model_FCNN import generate_model

import callback_custom
reload(callback_custom);


def pipeline_Kvalid(iK):



    num_classes = 11
    num_channel = 2

    # K-fold validation (K=5)
    #iK = 1
    n_training = 18
    n_test = 2

    idxs_test = list(range(1+n_test*(iK-1),1+n_test*iK))
    idxs_training = sorted(list(set(range(1,1+n_training+n_test))-set(idxs_test)))

    patience = 5
    model_filename = 'models/iK{}_outrun_step_{}.h5'.format(iK, '{}')
    csv_filename = 'log/iK{}_outrun_step_{}.cvs'.format(iK, '{}')
    output_filename = 'output/iK{}_result.cvs'.format(iK)

    nb_epoch = 40
    validation_split = 0.10
    monitor = 'val_loss'#'val_categorical_accuracy'

    class_mapper = {0:0}
    class_mapper.update({ i+1:i for i in range(1, 1+10) })
    class_mapper_inv = {0:0}
    class_mapper_inv.update({ i:i+1 for i in range(1, 1+10) })

    matrix_size = (160, 220, 48)

    extraction_step = (3, 3, 3)
    #extraction_step = (5, 5, 3)

    extraction_step_ft = (6, 6, 3)

    segment_size = (27, 27, 21)
    core_size = (9, 9, 3)




    # # Architecture

    # # 1. Initial segmentation

    # ## 1.1 Read data

    # In[52]:


    QSM_train = np.empty(((n_training,) + matrix_size), dtype=precision_global)
    MAG_train = np.empty(((n_training,) + matrix_size), dtype=precision_global)
    R2S_train = np.empty(((n_training,) + matrix_size), dtype=precision_global)
    label_train = np.empty(((n_training,) + matrix_size), dtype=precision_global)
    for i, case_idx in enumerate(idxs_training):
        QSM_train[i, :, :, :] = read_data(case_idx, 'QSM')
        MAG_train[i, :, :, :] = read_data(case_idx, 'MAG')
        R2S_train[i, :, :, :] = read_data(case_idx, 'R2S')
        label_train[i, :, :, :] = read_data(case_idx, 'label')


    # In[53]:


    #data_train = np.stack((QSM_train, MAG_train, R2S_train), axis = 1)
    #data_train = np.stack((QSM_train, R2S_train), axis = 1)
    #data_train = np.stack((QSM_train,), axis = 1)
    data_train = np.stack((QSM_train,), axis = 1)
    if num_channel > 1:
        data_train = np.stack((QSM_train, R2S_train), axis = 1)
    if num_channel > 2:
        data_train = np.stack((QSM_train, MAG_train, R2S_train), axis = 1)


    # In[54]:


    QSM_test = np.empty(((n_test,) + matrix_size), dtype=precision_global)
    MAG_test = np.empty(((n_test,) + matrix_size), dtype=precision_global)
    R2S_test = np.empty(((n_test,) + matrix_size), dtype=precision_global)
    label_test = np.empty(((n_test,) + matrix_size), dtype=precision_global)
    for i, case_idx in enumerate(idxs_test):
        QSM_test[i, :, :, :] = read_data(case_idx, 'QSM')
        MAG_test[i, :, :, :] = read_data(case_idx, 'MAG')
        R2S_test[i, :, :, :] = read_data(case_idx, 'R2S')
        label_test[i, :, :, :] = read_data(case_idx, 'label')


    # In[55]:


    #data_test = np.stack((QSM_test, MAG_test, R2S_test), axis = 1)
    #data_test = np.stack((QSM_test, R2S_test), axis = 1)
    #data_test = np.stack((QSM_test,), axis = 1)
    data_test = np.stack((QSM_test,), axis = 1)
    if num_channel > 1:
        data_test = np.stack((QSM_test, R2S_test), axis = 1)
    if num_channel > 2:
        data_test = np.stack((QSM_test, MAG_test, R2S_test), axis = 1)


    # ## 1.2 Pre-processing

    # In[56]:


    ## Intensity normalisation (zero mean and unit variance)
    input_mean = 127.0
    input_std = 64.0#128.0
    data_train = (data_train - input_mean) / input_std
    data_test = (data_test - input_mean) / input_std

    # Map class label
    tmp = np.copy(label_train)
    for class_idx in class_mapper:
        label_train[tmp == class_idx] = class_mapper[class_idx]
    tmp = np.copy(label_test)
    for class_idx in class_mapper:
        label_test[tmp == class_idx] = class_mapper[class_idx]
    del tmp


    # In[57]:


    label_train.max()


    # In[58]:


    plots(np.squeeze(label_train[0,:,:,[29,25]]), scale = (0, 10))


    # ## 1.3 Data preparation

    # In[59]:


    # In[65]:


    len_patch = extract_patches(read_data(1, 'QSM'), patch_shape=segment_size, extraction_step=(9, 9, 3)).shape[0]
    len_patch


    # In[66]:


    # Load best model
    model = generate_model(num_classes, num_channel, segment_size, core_size)
    model.load_weights(model_filename.format('1'))


    # In[81]:



    # In[82]:


    segmentations_test = []

    for i_case, case_idx in enumerate(idxs_test):

        print(case_idx)
        input_test = data_test[i_case, :, :, :, :]

        x_test = np.zeros((len_patch, num_channel,) + segment_size, dtype=precision_global)
        for i_channel in range(num_channel):
            x_test[:, i_channel, :, :, :] = extract_patches(input_test[i_channel], patch_shape=segment_size, extraction_step=(9, 9, 3))

        pred = model.predict(x_test, verbose=1)
        pred_classes = np.argmax(pred, axis=2)
        pred_classes = pred_classes.reshape((len(pred_classes), 9, 9, 3))
        segmentation = reconstruct_volume(pred_classes, matrix_size)
        #segmentation = reconstruct_volume_majority(pred_classes, matrix_size, extraction_step=(3, 3, 3))

        segmentations_test = segmentations_test + [segmentation]

    segmentations_test = np.stack(segmentations_test, axis=0)


    # In[83]:


    segmentations_test.max()


    # In[84]:


    plots(np.squeeze(label_test[:,:,:,23]), rows=1, scale = (0, 10))


    # In[85]:


    plots(np.squeeze(segmentations_test[:,:,:,23]), rows=1, scale = (0, 10))


    # # 3 Post-processing

    # ## 3.1 Pick the largest connected component for each class

    # In[87]:


    for i_case, case_idx in enumerate(idxs_test):
        segmentation = np.squeeze(segmentations_test[i_case,:,:,:]);
        tmp = np.zeros(segmentation.shape, dtype=segmentation.dtype)

        for class_idx in class_mapper_inv :
            mask = (segmentation == class_idx)

            if class_idx != 0 and mask.sum() > 0:
                labeled_mask, num_cc = ndimage.label(mask)
                largest_cc_mask = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))

                tmp[largest_cc_mask == 1] = class_idx

        segmentations_test[i_case,:,:,:] = tmp


    # ## 3.2 Save it 

    # In[88]:


    for i_case, case_idx in enumerate(idxs_test):
        print(case_idx)

        segmentation = np.copy(np.squeeze(segmentations_test[i_case,:,:,:]))

        tmp = np.copy(segmentation)
        for class_idx in class_mapper_inv:
            segmentation[tmp == class_idx] = class_mapper_inv[class_idx]
        del tmp

        save_data(segmentation, case_idx, 'label')    

    print("Done with Step 3")


    # ## 3.3 Calculate metric 

    # In[104]:


    orig_stdout = sys.stdout
    f = open(output_filename, 'w')
    sys.stdout = f


    # In[105]:


    def calc_dice(m1, m2):
        return 2*((m1==1) & (m2==1)).sum()/((m1==1).sum() + (m2==1).sum())


    # In[106]:


    for i_case, case_idx in enumerate(idxs_test):
        print(case_idx, end='\t')
        print('{:.4f}'.format(accuracy_score(label_test[i_case,:,:,:].flat, segmentations_test[i_case,:,:,:].flat)), end='\t')
        for class_idx in class_mapper_inv:
            mask = (np.squeeze(segmentations_test[i_case,:,:,:]) == class_idx)
            if class_idx != 0 and mask.sum() > 0:
                print('{:.4f}'.format(precision_score(label_test[i_case,:,:,:][mask], segmentations_test[i_case,:,:,:][mask], average='micro')), end='\t')
            else:
                print('N/A', end='\t')
        print()


    # In[107]:


    for i_case, case_idx in enumerate(idxs_test):
        print(case_idx, end='\t')
        for class_idx in class_mapper_inv:
            mask = (np.squeeze(segmentations_test[i_case,:,:,:]) == class_idx)
            if class_idx != 0 and mask.sum() > 0:
                print('{:.4f}'.format(calc_dice((label_test[i_case,:,:,:]==class_idx).flat, (segmentations_test[i_case,:,:,:]==class_idx).flat)), end='\t')
            else:
                print(0, end='\t')
        print()


    # In[108]:


    sys.stdout = orig_stdout
    f.close()

