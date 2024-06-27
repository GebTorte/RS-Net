#!/usr/bin/env python
# Needed to set seed for random generators for making reproducible experiments
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(1)

import datetime
import os
import sys
import math
import os.path
import random
import threading
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.backend import binary_crossentropy, sparse_categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler
from keras.utils import Sequence
from keras.utils import to_categorical
# from tensorflow.metrics import SparseCategoricalAccuracy

sys.path.insert(0, '../')
from srcv2_2.utils import extract_collapsed_cls, extract_cls_mask, image_normalizer, get_cls, shrink_cls_mask_to_indices, replace_fill_values


def swish(x):
    return (K.sigmoid(x) * x)

@keras.saving.register_keras_serializable()
def jaccard_coef(y_true, y_pred):
    """
    Calculates the Jaccard index
    """
    # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])  # Sum the product in all axes

    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])  # Sum the sum in all axes

    jac = (intersection + smooth) / (sum_ - intersection + smooth)  # Calc jaccard

    return K.mean(jac)

@keras.saving.register_keras_serializable()
def jaccard_coef_thresholded(y_true, y_pred):
    """
    Calculates the binarized Jaccard index
    """
    # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
    smooth = 1e-12

    # Round to 0 or 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    # Calculate Jaccard index
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

@keras.saving.register_keras_serializable()
def jaccard_coef_loss(y_true, y_pred):
    """
    Calculates the loss as a function of the Jaccard index and binary crossentropy
    """
    # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)

#@keras.saving.register_keras_serializable()
#def jaccard_coef_loss_sparse_categorical(y_true, y_pred):
#    """
#    Calculates the loss as a function of the Jaccard index and categporical crossentropy
#    """
#    # From https://github.com/ternaus/kaggle_dstl_submission/blob/master/src/unet_crops.py
#    return -K.log(jaccard_coef(y_true, y_pred)) + sparse_categorical_crossentropy(y_pred, y_true, ignore_class=0)

def cyclical_learning_rate_scheduler(epoch, lr, modulator = 7, epoch_cap=100):
    """
    quasi cyclical learning rate @ Smith 2015 
    """
    if epoch == 0 or epoch > epoch_cap: # stop cycling and stick with current lr
        return lr
    
    mod = epoch % modulator
    if mod == 0:
        return lr * modulator # set lr back to beginning lr
    original_lr = lr * mod
    return original_lr / (mod + 1)

def cyclical_learning_rate_scheduler_factorial(epoch, lr, modulator = 7, epoch_cap=100):
    """
    quasi cyclical learning rate @ Smith 2015 
    """
    if epoch == 0 or epoch > epoch_cap: # stop cycling and stick with current lr
        return lr
    
    mod = epoch % modulator
    if mod == 0:
        return lr * math.factorial(modulator) # set lr back to beginning lr
    return lr / (mod + 1)

def return_factor(_):
    return 2

def fib_iter(n):
    """
    @ https://stackoverflow.com/a/32215743/24005249
    """
    a, b = 0, 1
    for i in range(n):
        a, b = b, a + b
    return a

def step_learning_rate_scheduler(epoch, lr, epoch_step=10, divisor=2):
    if epoch % epoch_step == epoch_step-1:
        return lr/divisor
    return lr

def round_learning_rate_scheduler(epoch, lr, modulator=6, epoch_cap=100, divifactsor=1.5):
    """
    Note: modulator has to be of even.

    cyclical learning rate @ Smith 2015 
    """
    if epoch > epoch_cap or epoch==0: # or epoch%modulator==0: # stop cycling / @ middle of cycle and stick with current lr
        return lr
    
    epoch = epoch - 1 # reset mod position to start @ zero
    
    up = False
    signed_mod = epoch % modulator  -  modulator / 2

    if signed_mod >= 0:
        up = not up # switch every modulator/2 epochs
        
    if up:
        return lr * divifactsor
    else: # down
        return lr / divifactsor

def learning_rate_scheduler(epoch, lr, epoch_cap=15):
    if epoch > epoch_cap: # reduce lr exponentially
        return lr * tf.math.exp(-0.1)
    return lr

def custom_learning_rate_scheduler(epoch, lr, epoch_cap=1, exp=-0.7):
    epsilon = 1e-10
    if lr > epsilon: # dont drop below epsilon
        if epoch >= epoch_cap: # reduce lr exponentially
            return lr * tf.math.exp(exp) # maybe drop more strongly
        return lr
    return epsilon

def custom_learning_rate_scheduler_v2(epoch, lr, epoch_cap=1):
    epsilon = 1e-10
    default = 1e-7
    if lr > epsilon: # dont drop below epsilon
        if epoch >= epoch_cap: # reduce lr exponentially
            return default
        return lr
    return epsilon

def custom_defined_learning_rate_scheduler(epoch, lr, exp=-0.5):
    epoch_steps = [6, 13, 20]
    epsilon = 1e-10
    if lr > epsilon: # dont drop below epsilon
        if epoch in epoch_steps: # reduce lr exponentially
            return lr * tf.math.exp(exp)
        return lr
    return epsilon

@keras.saving.register_keras_serializable()
def get_catgorical_callbacks(params):
                
    categorical_model_checkpoint = ModelCheckpoint(params.project_path + f'models/Unet/{params.modelID}.keras',
                                       monitor='val_categorical_accuracy',
                                       save_weights_only=False,
                                       save_best_only=params.save_best_only)

    categorical_model_weights_checkpoint = ModelCheckpoint(params.project_path + f'models/Unet/{params.modelID}.h5',
                                       monitor='val_categorical_accuracy',
                                       save_weights_only=True,
                                       save_best_only=params.save_best_only)

    categorical_early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=params.early_patience, verbose=2)
    return categorical_model_checkpoint, categorical_model_weights_checkpoint, categorical_early_stopping


@keras.saving.register_keras_serializable()
def get_sparse_catgorical_callbacks(params):
            
    sparse_model_checkpoint = ModelCheckpoint(params.project_path + f'models/Unet/{params.modelID}.keras',
                                       monitor='val_sparse_categorical_accuracy',
                                       save_weights_only=False,
                                       save_best_only=params.save_best_only)

    sparse_model_weights_checkpoint = ModelCheckpoint(params.project_path + f'models/Unet/{params.modelID}.h5',
                                       monitor='val_sparse_categorical_accuracy',
                                       save_weights_only=True,
                                       save_best_only=params.save_best_only)

    sparse_early_stopping = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=params.early_patience, verbose=2)

    sparse_reduce_lr = ReduceLROnPlateau(factor=0.2, patience=params.plateau_patience, verbose=2, min_lr=1e-10)


    return sparse_model_checkpoint, sparse_model_weights_checkpoint,  sparse_early_stopping

@keras.saving.register_keras_serializable()
def get_callbacks(params):
    # Must use save_weights_only=True in model_checkpoint (BUG: https://github.com/fchollet/keras/issues/8123)
    model_checkpoint = ModelCheckpoint(params.project_path + 'models/Unet/unet_tmp.hdf5',
                                       monitor='val_acc',
                                       save_weights_only=True,
                                       save_best_only=params.save_best_only)

    model_checkpoint_saving = ModelCheckpoint(params.project_path + f'models/Unet/{params.modelID}.keras',
                                       monitor='val_acc',
                                       save_weights_only=False,
                                       save_best_only=params.save_best_only)

    early_stopping = EarlyStopping(monitor='val_acc', patience=100, verbose=2)

    tensorboard = TensorBoard(log_dir=params.project_path + f"reports/Unet/tensorboard/{params.modelID}",
                              write_graph=True,
                              write_images=True)

    csv_logger = CSVLogger(params.project_path + 'reports/Unet/csvlogger/' + params.modelID + '.log')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=2,
                                  patience=params.plateau_patience, min_lr=1e-10)
    
    cyclical_lr_scheduler = LearningRateScheduler(cyclical_learning_rate_scheduler, verbose=1)
    factorial_cyclical_lr_scheduler = LearningRateScheduler(cyclical_learning_rate_scheduler_factorial, verbose=1)
    round_cyclical_learning_rate_scheduler = LearningRateScheduler(round_learning_rate_scheduler, verbose=1)
    
    custom_1e2_learning_rate = LearningRateScheduler(custom_learning_rate_scheduler,verbose=1)


    return csv_logger, model_checkpoint, model_checkpoint_saving, reduce_lr, tensorboard, early_stopping, cyclical_lr_scheduler, factorial_cyclical_lr_scheduler, round_cyclical_learning_rate_scheduler, custom_1e2_learning_rate

class ImageSequence(Sequence):
    def __init__(self, params, shuffle, seed, augment_data, validation_generator=False):
        self.shuffle = shuffle

        # Find the number of classes and bands
        if params.collapse_cls:
            self.n_cls = 1
        else:
            self.n_cls = np.size(params.cls)
        self.n_bands = np.size(params.bands)
        # Load the names of the numpy files, each containing one patch
        if validation_generator:
            self.path = params.project_path + "data/processed/val/"
        else:
            self.path = params.project_path + "data/processed/train/" 
        self.x_files = sorted(os.listdir(self.path + "img/"))  # os.listdir loads in arbitrary order, hence use sorted()
        self.x_files = [f for f in self.x_files if '.npy' in f]  # Only use .npy files (e.g. avoid .gitkeep)
        self.y_files = sorted(os.listdir(self.path + "mask/"))  # os.listdir loads in arbitrary order, hence use sorted()
        self.y_files = [f for f in self.y_files if '.npy' in f]  # Only use .npy files (e.g. avoid .gitkeep)

        # Only train on the products as specified in the data split if using k-fold CV
        if params.split_dataset:
            for product in params.test_tiles[1]:
                self.x_files = [f for f in self.x_files if product[:-9] not in f]
                self.y_files = [f for f in self.y_files if product[:-9] not in f]

        # Create random generator used for shuffling files
        self.random = random.Random()
        self.tf_rng = tf.random.Generator.from_seed(seed)
        # Shuffle the patches
        if shuffle:
            self.random.seed(seed)
            self.random.shuffle(self.x_files)
            self.random.seed(seed)
            self.random.shuffle(self.y_files)

        self.batch_size = params.batch_size

        # Create placeholders
        self.x_all_bands = np.zeros((params.batch_size, params.patch_size, params.patch_size, 10), dtype=np.float32)
        self.x = np.zeros((params.batch_size, params.patch_size, params.patch_size, np.size(params.bands)), dtype=np.float32)

        self.clip_pixels = np.int32(params.overlap/2)
        # will change depending on actual overlap (v2 or v1 patch)

        self.y = np.zeros((params.batch_size, params.patch_size - 2*self.clip_pixels, params.patch_size-2*self.clip_pixels, 1), dtype=np.float32)

        # Load the params object for the normalizer function (not nice!)
        self.params = params

        # Convert class names to the actual integers in the masks (convert e.g. 'cloud' to 255 for Landsat8)
        #if self.params.loss_func == "sparse_categorical_crossentropy":
        #    # cls have then been provided as int already
        #    self.cls = self.params.cls 
        #else:

        #provide the cls as integers
        self.cls = get_cls(self.params.satellite, self.params.train_dataset, self.params.cls)

        # Augment the data
        self.augment_data = augment_data

    def __len__(self):
        return int(np.ceil(len(self.x_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_filenames = self.x_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_filenames = self.y_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        for i, filename in enumerate(batch_x_filenames):
            # Load all bands
            self.x_all_bands[i, :, :, :] = np.load(self.path + "img/" + filename)

            # Extract the wanted bands
            for j, b in enumerate(self.params.bands):
                if b == 1:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 0]
                elif b == 2:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 1]
                elif b == 3:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 2]
                elif b == 4:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 3]
                elif b == 5:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 4]
                elif b == 6:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 5]
                elif b == 7:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 6]
                elif b == 8:
                    raise ValueError('Band 8 (pan-chromatic band) cannot be included')
                elif b == 9:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 7]
                elif b == 10:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 8]
                elif b == 11:
                    self.x[i, :, :, j] = self.x_all_bands[i, :, :, 9]

            # Normalize
            self.x[i, :, :, :] = image_normalizer(self.x[i, :, :, :], self.params, self.params.norm_method)

        for i, filename in enumerate(batch_y_filenames):
            # Load the masks
            mask = np.load(self.path + "mask/" + filename)

            # for categorical, preplace 'fill' pxl with current most occuring pixel class.
            if self.params.loss_func == "sparse_categorical_crossentropy" or self.params.loss_func == "categorical_crossentropy":
                if self.params.replace_fill_values:
                    mask = replace_fill_values(self.params, self.params.train_dataset, mask, fill_with_value=self.params.dataset_fill_cls)

                # normalize the (biome_gt) mask values to range(0, len(cls)-1) 
                # loss will only compile then
                # do this after replacing fill values!
                mask = shrink_cls_mask_to_indices(self.params, self.params.train_dataset, self.params.cls, mask)

                assert type(mask) == np.ndarray

            # Create the binary masks
            if self.params.collapse_cls:
                mask = extract_collapsed_cls(mask, self.cls)

            # Save the (binary) mask (cropped)
            self.y[i, :, :, :] = mask[self.clip_pixels:self.params.patch_size - self.clip_pixels,
                                            self.clip_pixels:self.params.patch_size - self.clip_pixels,
                                            :]
        
        if self.augment_data:
            if self.random.randint(0, 1):
                np.flip(self.x, axis=1)
                np.flip(self.y, axis=1)

            if self.random.randint(0, 1):
                np.flip(self.x, axis=2)
                np.flip(self.y, axis=2)

        # assert training data is valid and does not introduce nan to model
        assert not np.any(np.isnan(self.x))
        assert not np.any(np.isnan(self.y))

        if self.params.loss_func == "categorical_crossentropy": 
            # should theoretically work
            # but categorical with these batch sizes exhausts the gpu resources -> reduce batch size
            batch_x = tf.convert_to_tensor(self.x)
            batch_y = keras.utils.to_categorical(np.int64(self.y))# tf.convert_to_tensor(keras.utils.to_categorical(np.int32(self.y)))
        elif self.params.loss_func == "sparse_categorical_crossentropy":
            batch_x = tf.convert_to_tensor(self.x) # tf.convert_to_tensor()
            batch_y = tf.convert_to_tensor(np.squeeze(np.int32(self.y), axis=-1)) # tf.convert_to_tensor(np.int32(self.y))# # cast to int
        else:# binary crossentropy
            batch_x = self.x
            batch_y = self.y
        
        return batch_x, batch_y

    def on_epoch_end(self):
        pass
        # Shuffle the patches
        # USE Shuffle from model.fit() for this
        #if self.shuffle:
        #    rng_seed = int(self.tf_rng.uniform(shape=(), minval=0, maxval=255, dtype=tf.int32))
        #    self.random.seed(rng_seed)
        #    self.random.shuffle(self.x_files)
        #    self.random.seed(rng_seed)
        #    self.random.shuffle(self.y_files)