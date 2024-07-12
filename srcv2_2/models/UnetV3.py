#!/usr/bin/env python
# Needed to set seed for random generators for making reproducible experiments
import numpy.random

#set_seed(1)

import json
import numpy as np
import tensorflow as tf
import tensorflow.distribute
import tensorflow.keras as keras
import keras.utils
from keras.models import Model
from keras import regularizers
from keras.metrics import Precision, Recall
from keras.initializers import GlorotNormal, HeNormal
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout, Cropping2D, Activation, BatchNormalization, LeakyReLU
from keras.optimizers.legacy import Adam, Nadam
from keras.optimizers import AdamW
from keras.utils import get_custom_objects  # To use swish activation function
#from tensorflow.keras.utils import multi_gpu_model
from ..utils import get_model_name, get_cls
from srcv2_2.models.model_utils import jaccard_coef, jaccard_coef_thresholded, jaccard_coef_loss, swish, get_callbacks, get_sparse_catgorical_callbacks, get_catgorical_callbacks,ImageSequence #, jaccard_coef_loss_sparse_categorical
from srcv2_2.models.params import HParams


class UnetV3(object):
    def __init__(self, params, model=None):
        # Seed for the random generators
        self.seed = 1

        self.params = params
        self.model = model

        self.strategy = None

        try:
            self.params.update(**self._load_params())
            print("Params have been loaded:", self.params.as_string())
        except:
            print("Params were not loaded.")

        # set these random generator seeds to ensure more comparable output. Otherwise train model multiple times and evaluate means
        if not self.params.random:
            #numpy.random.seed(1)
            #tf.random.set_seed(self.seed)
            keras.utils.set_random_seed(self.seed)

        # Find the model you would like
        self.model_name = get_model_name(self.params)

        # Find the number of classes and bands
        if self.params.collapse_cls:
            self.n_cls = 1
        else:
            self.n_cls = np.size(self.params.cls)
        self.n_bands = np.size(self.params.bands)

        # Create the model in keras, if not provided
        if model == None:
            # Try loading a saved model. get_model_name has to be unique
            try:
                self.model = keras.saving.load_model(self.params.project_path + 'models/Unet/' + self.params.modelID + '.keras') # get_model_name(self.params)
                print(f"Model {self.params.modelID} has been loaded.")

                return # dont create inference
            except:
                print("No model was loaded.")

            if self.params.num_gpus == 1:
                #try:
                #    gpu = tf.config.experimental.list_physical_devices("GPU")[0]
                #    tf.config.experimental.set_memory_growth(gpu, True)
                #    tf.config.experimental.set_virtual_device_configuration(
                #        gpu,
                #        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=self.params.gpu_limit)])
                #    #logical_gpus = tf.config.list_logical_devices('GPU')
                #    virtual_gpus = tf.config.experimental.list_logical_devices("GPU")
                #    gpu = virtual_gpus[0]
                #    print("GPU", str(gpu))
                #    print(virtual_gpus)
                #except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                #    print(e)
                #    quit()

                # define strategy with gpu limit
                #self.strategy = tf.distribute.OneDeviceStrategy(gpu.name)
                #with self.strategy.scope():
                #    self.model = self.__create_inference__()

                with tf.device("/GPU:0"):
                    self.model = self.__create_inference__()  # initialize the model

            else:
                print("ATTENTION: Only running on CPU")
                with tf.device("/cpu:0"):
                    self.model = self.__create_inference__()  # initialize the model on the CPU
            # deprecated -> use tf.distribute.MirroredStrategy().scope() for compiling and training on mulitple gpus instead
            # self.model = multi_gpu_model(self.model, gpus=self.params.num_gpus)  # Make it run on multiple GPUs
        else:
            print(f"Model {get_model_name(self.params)} has been provided in constructor.")
            return

    def __create_inference__(self):
        # Note about BN and dropout: https://stackoverflow.com/questions/46316687/how-to-include-batch-normalization-in-non-sequential-keras-model


        #model_v3 = self._load_model_v3()
        #model_v3.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))
        #print(model_v3.summary())

        #model_v3_128 = self._load_model_v3_128()
        #model_v3_128.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))
        #print(model_v3_128.summary())

        #model_v3_1024 = self._load_model_v3_1024()
        #model_v3_1024.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))
        #print(model_v3_1024.summary())

        #model_v3_1024_16 = self._load_model_v3_1024_16()
        #model_v3_1024_16.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))
        #print(model_v3_1024_16.summary())

        # rainio for segmentation
        # @ https://github.com/rklen/statistical_tests_for_CNNs/blob/main/stm.ipynb
        #model = self._load_rainio_deeper_unet_parametrized()
        #model.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))
        #print(model.summary())

        #model = self._load_rainio_deeper_unet_256()
        #model.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))
        #print(model.summary())
        
        #model = self._load_rainio_deeper_unet_v3_32_256(kernel_shape=(7,7))
        #model.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))
        #model.summary(print_fn=UnetV3.print_summary)
        #print(model.summary())


        #model = self._load_rainio_deeper_unet_v3_32_512()
        #model = self._load_rainio_deeper_unet_v3_32_1024()
        #model.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))
        #print(model.summary())

        #---------------------------------------------------------#
        # JEPPESEN 
        model = self._load_model_v2(kernel_shape=(7,7))
        model.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))
        model.summary(print_fn=UnetV3.print_summary)
        print(model.summary())

        return model
    
    @staticmethod
    def print_summary(s):
        with open("UnetV3_summary.html", "a") as f:
            f.write(s + "\n")
    
    def _load_rainio_deeper_unet_v3_8_256(self):
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1 if dropout1 != 0 else 0

        #Contraction path
        c0 = keras.layers.Conv2D(8, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg), padding='same')(inputs)
        c0 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c0) if self.params.use_batch_norm else c0
        c0 = keras.layers.Dropout(dropout1)(c0)
        c0 = keras.layers.Conv2D(8, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c0)
        c0 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c0) if self.params.use_batch_norm else c0
        p0 = keras.layers.MaxPooling2D((2, 2))(c0)

        c1 = keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg), padding='same')(p0)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        c1 = keras.layers.Dropout(dropout1)(c1)
        c1 = keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c1)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p1)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p4)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        c5 = keras.layers.Dropout(dropout2)(c5)
        c5 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c5)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5

        #Expansive path
        u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = keras.layers.concatenate([u6, c4])
        c6 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u6)
        c6 = keras.layers.Dropout(dropout1)(c6)
        c6 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c6)

        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = keras.layers.concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u9)
        c9 = Dropout(dropout1)(c9)
        c9 = keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c9)

        u10 = keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c9)
        u10 = keras.layers.concatenate([u10, c0], axis=3)
        c10 = keras.layers.Conv2D(8, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u10)
        c10 = Dropout(dropout1)(c10)
        c10 = keras.layers.Conv2D(8, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c10)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to (216,216) (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c10)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def _load_rainio_deeper_unet_v3_32_256(self, kernel_shape=(3,3)):
        """
        merge of rainio and jeppesen
        """
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1 if dropout1 != 0 else 0

        #Contraction path

        c2 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(inputs)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = keras.layers.Conv2D(256, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p4)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        c5 = keras.layers.Dropout(dropout2)(c5)
        c5 = keras.layers.Conv2D(256, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c5)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5

        #Expansive path

        u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = keras.layers.concatenate([u6, c4])
        c6 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u6)
        c6 = keras.layers.Dropout(dropout1)(c6)
        c6 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c6)

        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to (216,216) (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c8)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def _load_rainio_deeper_unet_v3_32_x3_864(self):
        """
        merge of rainio and jeppesen
        """
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1 if dropout1 != 0 else 0

        #Contraction path

        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(inputs)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(96, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(96, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(288, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(288, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = keras.layers.Conv2D(864, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p4)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        c5 = keras.layers.Dropout(dropout2)(c5)
        c5 = keras.layers.Conv2D(864, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c5)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5

        #Expansive path
        u6 = keras.layers.Conv2DTranspose(288, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = keras.layers.concatenate([u6, c4])
        c6 = keras.layers.Conv2D(288, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u6)
        c6 = keras.layers.Dropout(dropout1)(c6)
        c6 = keras.layers.Conv2D(288, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c6)

        u7 = keras.layers.Conv2DTranspose(96, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(96, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(96, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to (216,216) (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c8)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    

    def _load_rainio_deeper_unet_v3_32_512(self):
        """
        merge of rainio and jeppesen
        """
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1 if dropout1 != 0 else 0

        #Contraction path

        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(inputs)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p4)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        c5 = keras.layers.Dropout(dropout2)(c5)
        c5 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c5)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        p5 = keras.layers.MaxPooling2D((2, 2))(c5)

        c51 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p5)
        c51 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c51) if self.params.use_batch_norm else c51
        c51 = keras.layers.Dropout(dropout2)(c51)
        c51 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c51)
        c51 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c51) if self.params.use_batch_norm else c51

        #Expansive path
        u52 = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c51)
        u52 = keras.layers.concatenate([u52, c5])
        c52 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u52)
        c52 = keras.layers.Dropout(dropout1)(c52)
        c52 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c52)

        u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c52)
        u6 = keras.layers.concatenate([u6, c4])
        c6 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u6)
        c6 = keras.layers.Dropout(dropout1)(c6)
        c6 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c6)

        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to (216,216) (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c8)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def _load_rainio_deeper_unet_v3_32_1024(self):
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1 if dropout1 != 0 else 0

        #Contraction path
        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(inputs)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p4)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        c5 = keras.layers.Dropout(dropout2)(c5)
        c5 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c5)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        p5 = keras.layers.MaxPooling2D((2, 2))(c5)

        c51 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p5)
        c51 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c51) if self.params.use_batch_norm else c51
        c51 = keras.layers.Dropout(dropout2)(c51)
        c51 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c51)
        c51 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c51) if self.params.use_batch_norm else c51
        p51 = keras.layers.MaxPooling2D((2, 2))(c51)

        c52 = keras.layers.Conv2D(1024, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p51)
        c52 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c52) if self.params.use_batch_norm else c52
        c52 = keras.layers.Dropout(dropout2)(c52)
        c52 = keras.layers.Conv2D(1024, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c52)
        c52 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c52) if self.params.use_batch_norm else c52

        #Expansive path
        u62 = keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c52)
        u62 = keras.layers.concatenate([u62, c51])
        c62 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u62)
        c62 = keras.layers.Dropout(dropout1)(c62)
        c62 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c62)

        u61 = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c62)
        u61 = keras.layers.concatenate([u61, c5])
        c61 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u61)
        c61 = keras.layers.Dropout(dropout1)(c61)
        c61 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c61)

        u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c61)
        u6 = keras.layers.concatenate([u6, c4])
        c6 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u6)
        c6 = keras.layers.Dropout(dropout1)(c6)
        c6 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c6)

        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to (216,216) (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c8)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def _load_rainio_deeper_unet_v3_16_1024(self):
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1 if dropout1 != 0 else 0

        #Contraction path
        c1 = keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg), padding='same')(inputs)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        c1 = keras.layers.Dropout(dropout1)(c1)
        c1 = keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c1)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p1)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p4)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        c5 = keras.layers.Dropout(dropout2)(c5)
        c5 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c5)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        p5 = keras.layers.MaxPooling2D((2, 2))(c5)

        c51 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p5)
        c51 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c51) if self.params.use_batch_norm else c51
        c51 = keras.layers.Dropout(dropout2)(c51)
        c51 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c51)
        c51 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c51) if self.params.use_batch_norm else c51
        p51 = keras.layers.MaxPooling2D((2, 2))(c51)

        c52 = keras.layers.Conv2D(1024, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p51)
        c52 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c52) if self.params.use_batch_norm else c52
        c52 = keras.layers.Dropout(dropout2)(c52)
        c52 = keras.layers.Conv2D(1024, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c52)
        c52 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c52) if self.params.use_batch_norm else c52

        #Expansive path
        u62 = keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c52)
        u62 = keras.layers.concatenate([u62, c51])
        c62 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u62)
        c62 = keras.layers.Dropout(dropout1)(c62)
        c62 = keras.layers.Conv2D(512, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c62)

        u61 = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c62)
        u61 = keras.layers.concatenate([u61, c5])
        c61 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u61)
        c61 = keras.layers.Dropout(dropout1)(c61)
        c61 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c61)

        u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c61)
        u6 = keras.layers.concatenate([u6, c4])
        c6 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u6)
        c6 = keras.layers.Dropout(dropout1)(c6)
        c6 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c6)

        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = keras.layers.concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u9)
        c9 = Dropout(dropout1)(c9)
        c9 = keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c9)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to (216,216) (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop91 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c9)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop91)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def _load_rainio_deeper_unet_v3_16_512(self, kernel_shape=(3,3)):
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1 if dropout1 != 0 else 0

        #Contraction path
        c1 = keras.layers.Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg), padding='same')(inputs)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        c1 = keras.layers.Dropout(dropout1)(c1)
        c1 = keras.layers.Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c1)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p1)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = keras.layers.Conv2D(256, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p4)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        c5 = keras.layers.Dropout(dropout2)(c5)
        c5 = keras.layers.Conv2D(256, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c5)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        p5 = keras.layers.MaxPooling2D((2, 2))(c5)

        c51 = keras.layers.Conv2D(512, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p5)
        c51 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c51) if self.params.use_batch_norm else c51
        c51 = keras.layers.Dropout(dropout2)(c51)
        c51 = keras.layers.Conv2D(512, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c51)
        c51 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c51) if self.params.use_batch_norm else c51

        #Expansive path
        u52 = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c51)
        u52 = keras.layers.concatenate([u52, c5])
        c52 = keras.layers.Conv2D(256, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u52)
        c52 = keras.layers.Dropout(dropout1)(c52)
        c52 = keras.layers.Conv2D(256, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c52)

        u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c52)
        u6 = keras.layers.concatenate([u6, c4])
        c6 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u6)
        c6 = keras.layers.Dropout(dropout1)(c6)
        c6 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c6)

        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = keras.layers.concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u9)
        c9 = Dropout(dropout1)(c9)
        c9 = keras.layers.Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c9)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to (216,216) (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop91 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c9)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop91)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def _load_rainio_deeper_unet_v3_16_256(self, kernel_shape=(3,3)):
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1 if dropout1 != 0 else 0

        #Contraction path
        c1 = keras.layers.Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg), padding='same')(inputs)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        c1 = keras.layers.Dropout(dropout1)(c1)
        c1 = keras.layers.Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c1)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p1)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = keras.layers.Conv2D(256, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p4)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5
        c5 = keras.layers.Dropout(dropout2)(c5)
        c5 = keras.layers.Conv2D(256, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c5)
        c5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c5) if self.params.use_batch_norm else c5

        #Expansive path
        u6 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = keras.layers.concatenate([u6, c4])
        c6 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u6)
        c6 = keras.layers.Dropout(dropout1)(c6)
        c6 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c6)

        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = keras.layers.concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u9)
        c9 = Dropout(dropout1)(c9)
        c9 = keras.layers.Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c9)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to (216,216) (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop91 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c9)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop91)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def _load_rainio_deeper_unet_v3(self, kernel_shape=(3,3)):
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands), name="U-net-rainio-deeper-v2-16-128")
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1

        #Contraction path
        c1 = keras.layers.Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg), padding='same')(inputs)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        c1 = keras.layers.Dropout(dropout1)(c1)
        c1 = keras.layers.Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c1)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p1)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(128, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4

        #Expansive path
        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(64, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(32, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = keras.layers.concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u9)
        c9 = Dropout(dropout1)(c9)
        c9 = keras.layers.Conv2D(16, kernel_shape, activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c9)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to 216,216 (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop91 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c9)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop91)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
    
    def _load_rainio_deeper_unet_256(self):
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        activation = self.params.activation_func
        kernel_init = self.params.initialization
        #kernel_regularizer=regularizers.l2(self.params.L2reg)

        dropout1 = self.params.dropout
        dropout2 = dropout1 + .1

        #Contraction path
        c1 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg), padding='same')(inputs)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        c1 = keras.layers.Dropout(dropout1)(c1)
        c1 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c1)
        c1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c1) if self.params.use_batch_norm else c1
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p1)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        c2 = keras.layers.Dropout(dropout1)(c2)
        c2 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c2)
        c2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c2) if self.params.use_batch_norm else c2
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p2)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        c3 = keras.layers.Dropout(dropout2)(c3)
        c3 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c3)
        c3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c3) if self.params.use_batch_norm else c3
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(p3)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4
        c4 = keras.layers.Dropout(dropout2)(c4)
        c4 = keras.layers.Conv2D(256, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c4)
        c4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(c4) if self.params.use_batch_norm else c4

        #Expansive path
        u7 = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u7)
        c7 = keras.layers.Dropout(dropout1)(c7)
        c7 = keras.layers.Conv2D(128, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u8)
        c8 = keras.layers.Dropout(dropout1)(c8)
        c8 = keras.layers.Conv2D(64, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c8)

        u9 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = keras.layers.concatenate([u9, c1], axis=3)
        c9 = Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(u9)
        c9 = Dropout(dropout1)(c9)
        c9 = keras.layers.Conv2D(32, (3, 3), activation=activation, kernel_initializer=kernel_init, kernel_regularizer=regularizers.l2(self.params.L2reg),padding='same')(c9)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to 216,216 (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)   # Only used for input in Cropping2D function on next line
        crop91 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c9)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation=self.params.last_layer_activation_func)(crop91)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def _load_rainio_deeper_unet(self):
            
        #U-Net model
        inputs = keras.layers.Input(shape=(self.params.patch_size,self.params.patch_size,self.n_bands))
        # , batch_size=self.params.batch_size

        #Contraction path
        c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = keras.layers.Dropout(0.1)(c1)
        c1 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = keras.layers.Dropout(0.1)(c2)
        c2 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = keras.layers.Dropout(0.2)(c3)
        c3 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = keras.layers.Dropout(0.2)(c4)
        c4 = keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

        #Expansive path
        u7 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
        u7 = keras.layers.concatenate([u7, c3])
        c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = keras.layers.Dropout(0.1)(c7)
        c7 = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = keras.layers.concatenate([u8, c2])
        c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = keras.layers.Dropout(0.1)(c8)
        c8 = keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = keras.layers.concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        # ------------------------------------------------------------------------------------------------------------------------------------
        # clipping to 216,216 (@ 40 overlap)
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop91 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(c9)
        # ------------------------------------------------------------------------------------------------------------------------------------

        outputs = keras.layers.Conv2D(self.n_cls, (1,1), activation="softmax")(crop91) #sigmoid or linear

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def _load_model_v2_1024_16(self):
        """
        !work in progress!
        
        adding a 1024 layer
        """
        activation_func = self.params.activation_func

        inputs = Input((self.params.patch_size, self.params.patch_size, self.n_bands))
        # -----------------------------------------------------------------------
        conv0 = Conv2D(16, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(inputs)
        conv0 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv0) if self.params.use_batch_norm else conv0
        conv0 = Conv2D(16, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv0)
        conv0 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv0) if self.params.use_batch_norm else conv0
        pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
        # -----------------------------------------------------------------------
        conv1 = Conv2D(32, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool0)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        conv1 = Conv2D(32, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv1)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # -----------------------------------------------------------------------
        conv2 = Conv2D(64, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool1)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        conv2 = Conv2D(64, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv2)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # -----------------------------------------------------------------------
        conv3 = Conv2D(128, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool2)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        conv3 = Conv2D(128, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv3)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # -----------------------------------------------------------------------
        conv4 = Conv2D(256, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool3)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        conv4 = Conv2D(256, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv4)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        # -----------------------------------------------------------------------
        conv5 = Conv2D(512, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool4)
        conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        conv5 = Conv2D(512, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv5)
        conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        # -----------------------------------------------------------------------
        #up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
        # -----------------------------------------------------------------------
        conv52 = Conv2D(1024, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool5)
        conv52 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv52) if self.params.use_batch_norm else conv52
        conv52 = Conv2D(1024, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv52)
        conv52 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv52) if self.params.use_batch_norm else conv52
        # -----------------------------------------------------------------------
        up61 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv52), conv5])
        conv61 = Conv2D(512, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up61)
        conv61 = Dropout (self.params.dropout)(conv61) if not self.params.dropout_on_last_layer_only else conv61
        conv61 = Conv2D(512, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv61)
        conv61 = Dropout(self.params.dropout)(conv61) if not self.params.dropout_on_last_layer_only else conv61
        # -----------------------------------------------------------------------
        up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv61), conv4])
        conv6 = Conv2D(256, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up6)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        conv6 = Conv2D(256, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv6)
        conv6 = Dropout(self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        # -----------------------------------------------------------------------
        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up7)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        conv7 = Conv2D(128, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv7)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        # -----------------------------------------------------------------------
        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        conv8 = Conv2D(64, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        # -----------------------------------------------------------------------
        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up9)
        conv9 = Dropout(self.params.dropout)(conv9) if not self.params.dropout_on_last_layer_only else conv9
        conv9 = Conv2D(32, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv9)
        conv9 = Dropout(self.params.dropout)(conv9)
        # -----------------------------------------------------------------------
        up91 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv9), conv0])
        conv91 = Conv2D(16, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up91)
        conv91 = Dropout(self.params.dropout)(conv91) if not self.params.dropout_on_last_layer_only else conv91
        conv91 = Conv2D(16, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv91)
        conv91 = Dropout(self.params.dropout)(conv91)
        # -----------------------------------------------------------------------
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop91 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv91)
        # -----------------------------------------------------------------------
        # SIS: change to softmax for multi class prediction
        conv10 = Conv2D(self.n_cls, (1, 1), activation=self.params.last_layer_activation_func, 
                        kernel_initializer=self.params.initialization)(crop91)
        # -----------------------------------------------------------------------
        model = Model(inputs=inputs, outputs=conv10)

        return model
    
    def _load_model_v2_1024(self, kernel_shape=(3,3)):
        """
        !work in progress!
        
        adding a 1024 layer
        """
        activation_func = self.params.activation_func

        inputs = Input((self.params.patch_size, self.params.patch_size, self.n_bands))
        # -----------------------------------------------------------------------
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(inputs)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv1)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # -----------------------------------------------------------------------
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool1)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv2)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # -----------------------------------------------------------------------
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool2)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv3)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # -----------------------------------------------------------------------
        conv4 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool3)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        conv4 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv4)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        # -----------------------------------------------------------------------
        conv5 = Conv2D(512, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool4)
        conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        conv5 = Conv2D(512, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv5)
        conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        # -----------------------------------------------------------------------
        #up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
        # -----------------------------------------------------------------------
        conv52 = Conv2D(1024, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool5)
        conv52 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv52) if self.params.use_batch_norm else conv52
        conv52 = Conv2D(1024, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv52)
        conv52 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv52) if self.params.use_batch_norm else conv52
        # -----------------------------------------------------------------------
        up61 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv52), conv5])
        conv61 = Conv2D(512, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up61)
        conv61 = Dropout (self.params.dropout)(conv61) if not self.params.dropout_on_last_layer_only else conv61
        conv61 = Conv2D(512, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv61)
        conv61 = Dropout(self.params.dropout)(conv61) if not self.params.dropout_on_last_layer_only else conv61
        # -----------------------------------------------------------------------
        up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv61), conv4])
        conv6 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up6)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        conv6 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv6)
        conv6 = Dropout(self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        # -----------------------------------------------------------------------
        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up7)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv7)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        # -----------------------------------------------------------------------
        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        # -----------------------------------------------------------------------
        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up9)
        conv9 = Dropout(self.params.dropout)(conv9) if not self.params.dropout_on_last_layer_only else conv9
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv9)
        conv9 = Dropout(self.params.dropout)(conv9)
        # -----------------------------------------------------------------------
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv9)
        # -----------------------------------------------------------------------
        # SIS: change to softmax for multi class prediction
        conv10 = Conv2D(self.n_cls, (1, 1), activation=self.params.last_layer_activation_func, 
                        kernel_initializer=self.params.initialization)(crop9)
        # -----------------------------------------------------------------------
        model = Model(inputs=inputs, outputs=conv10)

        return model
    
    def _load_model_v2_256_16(self):
        activation_func = self.params.activation_func
        kernel_shape=(7,7)

        inputs = Input((self.params.patch_size, self.params.patch_size, self.n_bands))
        # -----------------------------------------------------------------------
        # conv0 with 16, (3,3) perhaps?
        conv0 = Conv2D(16, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(inputs)
        conv0 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv0) if self.params.use_batch_norm else conv0
        conv0 = Conv2D(16, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv0)
        conv0 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv0) if self.params.use_batch_norm else conv0
        pool0 = MaxPooling2D(pool_size=(2, 2))(conv0)
        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool0)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv1)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # -----------------------------------------------------------------------
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool1)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv2)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # -----------------------------------------------------------------------
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool2)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv3)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # -----------------------------------------------------------------------
        conv4 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool3)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        conv4 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv4)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        # -----------------------------------------------------------------------
        #conv5 = Conv2D(512, kernel_shape, activation=activation_func, padding='same',
        #               kernel_regularizer=regularizers.l2(self.params.L2reg),
        #               kernel_initializer=self.params.initialization)(pool4)
        #conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        #conv5 = Conv2D(512, kernel_shape, activation=activation_func, padding='same',
        #               kernel_regularizer=regularizers.l2(self.params.L2reg),
        #               kernel_initializer=self.params.initialization)(conv5)
        #conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        # -----------------------------------------------------------------------
        # up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])

        # -----------------------------------------------------------------------
        # add this instead maybe?
        # # up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv4), conv3])
        conv6 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv4)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        conv6 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv6)
        conv6 = Dropout(self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        # -----------------------------------------------------------------------
        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up7)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv7)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        # -----------------------------------------------------------------------
        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        # -----------------------------------------------------------------------
        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up9)
        conv9 = Dropout(self.params.dropout)(conv9) if not self.params.dropout_on_last_layer_only else conv9
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv9)
        conv9 = Dropout(self.params.dropout)(conv9)
        # -----------------------------------------------------------------------
        up91 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv9), conv0])
        conv91 = Conv2D(16, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up91)
        conv91 = Dropout(self.params.dropout)(conv91) if not self.params.dropout_on_last_layer_only else conv91
        conv91 = Conv2D(16, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv91)
        conv91 = Dropout(self.params.dropout)(conv91)
        # -----------------------------------------------------------------------
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop91 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv91)
        # -----------------------------------------------------------------------
        # SIS: change to softmax for multi class prediction
        conv10 = Conv2D(self.n_cls, (1, 1), activation=self.params.last_layer_activation_func, 
                        kernel_initializer=self.params.initialization)(crop91)
        # -----------------------------------------------------------------------
        model = Model(inputs=inputs, outputs=conv10)

        return model

    def _load_model_v2_256(self):
        activation_func = self.params.activation_func
        kernel_shape = (7,7)

        inputs = Input((self.params.patch_size, self.params.patch_size, self.n_bands))
        # -----------------------------------------------------------------------
        # conv0 with 16, (3,3) perhaps?
        # -----------------------------------------------------------------------
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(inputs)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv1)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # -----------------------------------------------------------------------
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool1)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv2)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # -----------------------------------------------------------------------
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool2)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv3)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # -----------------------------------------------------------------------
        conv4 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool3)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        conv4 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv4)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        # -----------------------------------------------------------------------

        conv6 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv4)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        conv6 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv6)
        conv6 = Dropout(self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        # -----------------------------------------------------------------------
        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up7)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv7)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        # -----------------------------------------------------------------------
        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        # -----------------------------------------------------------------------
        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up9)
        conv9 = Dropout(self.params.dropout)(conv9) if not self.params.dropout_on_last_layer_only else conv9
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv9)
        conv9 = Dropout(self.params.dropout)(conv9)
        # -----------------------------------------------------------------------
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv9)
        # -----------------------------------------------------------------------
        # SIS: change to softmax for multi class prediction
        conv10 = Conv2D(self.n_cls, (1, 1), activation=self.params.last_layer_activation_func, 
                        kernel_initializer=self.params.initialization)(crop9)
        # -----------------------------------------------------------------------
        model = Model(inputs=inputs, outputs=conv10)

        return model
    
    def _load_model_v2_128(self):
        activation_func = self.params.activation_func
        kernel_shape = (3,3)

        inputs = Input((self.params.patch_size, self.params.patch_size, self.n_bands))
        # -----------------------------------------------------------------------
        # conv0 with 16, (3,3) perhaps?
        # -----------------------------------------------------------------------
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(inputs)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv1)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # -----------------------------------------------------------------------
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool1)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv2)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # -----------------------------------------------------------------------
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool2)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv3)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        #pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv3)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv7)
        conv7 = Dropout(self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        # -----------------------------------------------------------------------
        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv8)
        conv8 = Dropout(self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        # -----------------------------------------------------------------------
        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up9)
        conv9 = Dropout(self.params.dropout)(conv9) if not self.params.dropout_on_last_layer_only else conv9
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv9)
        conv9 = Dropout(self.params.dropout)(conv9)
        # -----------------------------------------------------------------------
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv9)
        # -----------------------------------------------------------------------
        # SIS: change to softmax for multi class prediction
        conv10 = Conv2D(self.n_cls, (1, 1), activation=self.params.last_layer_activation_func, 
                        kernel_initializer=self.params.initialization)(crop9)
        # -----------------------------------------------------------------------
        model = Model(inputs=inputs, outputs=conv10)

        return model
    
    def _load_model_v2(self, kernel_shape=(3,3)):
        activation_func = self.params.activation_func

        inputs = Input((self.params.patch_size, self.params.patch_size, self.n_bands))
        # -----------------------------------------------------------------------
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(inputs)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        conv1 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv1)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # -----------------------------------------------------------------------
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool1)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        conv2 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv2)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # -----------------------------------------------------------------------
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool2)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        conv3 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv3)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # -----------------------------------------------------------------------
        conv4 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool3)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        conv4 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv4)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        # -----------------------------------------------------------------------
        conv5 = Conv2D(512, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(pool4)
        conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        conv5 = Conv2D(512, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv5)
        conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        # -----------------------------------------------------------------------
        up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up6)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        conv6 = Conv2D(256, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv6)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        # -----------------------------------------------------------------------
        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up7)
        conv7 = Dropout (self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        conv7 = Conv2D(128, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv7)
        conv7 = Dropout (self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        # -----------------------------------------------------------------------
        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up8)
        conv8 = Dropout (self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        conv8 = Conv2D(64, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv8)
        conv8 = Dropout (self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        # -----------------------------------------------------------------------
        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up9)
        conv9 = Dropout(self.params.dropout)(conv9) if not self.params.dropout_on_last_layer_only else conv9
        conv9 = Conv2D(32, kernel_shape, activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv9)
        conv9 = Dropout(self.params.dropout)(conv9)
        # -----------------------------------------------------------------------
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv9)
        # -----------------------------------------------------------------------
        # SIS: change to softmax for multi class prediction
        conv10 = Conv2D(self.n_cls, (1, 1), activation=self.params.last_layer_activation_func, 
                        kernel_initializer=self.params.initialization)(crop9)
        # -----------------------------------------------------------------------
        model = Model(inputs=inputs, outputs=conv10)

        return model

    def get_config(self):
        return {'seed': self.seed, 'params': self.params, 'n_cls': self.n_cls, 'n_bands': self.n_bands, 'model_config': self.model.get_config()}

    def train(self):
        if self.strategy:
            with self.strategy.scope():
                self._train()
        else:
            self._train()
    
    def _train(self):
        print()
        print(f"Model {self.params.modelID}: Training on params: ", self.params.as_string(delimiter="\n", skip_keys_list=["test_tiles", "project_path", "data_path", "toa_path"]))
        print()

        # Define callbacks
        csv_logger, model_checkpoint, model_checkpoint_saving, model_checkpoint_saving_loss, reduce_lr, tensorboard, early_stopping, round_cyclical_learning_rate_scheduler, cyclical_lr_callback = get_callbacks(self.params)
        used_callbacks = [csv_logger,  tensorboard]
        
        if self.params.reduce_lr:
            used_callbacks.append(reduce_lr)
        if self.params.lr_scheduler:
            used_callbacks.append(cyclical_lr_callback)
        if self.params.loss_func == "binary_crossentropy":
            if self.params.early_stopping:
                used_callbacks.append(early_stopping)
            used_callbacks.append(model_checkpoint)
            used_callbacks.append(model_checkpoint_saving)
        elif self.params.loss_func == "sparse_categorical_crossentropy":
            sparse_model_checkpoint, sparse_model_weights_checkpoint,  sparse_early_stopping = get_sparse_catgorical_callbacks(self.params)
            if self.params.early_stopping:
                used_callbacks.append(sparse_early_stopping)
            used_callbacks.append(sparse_model_checkpoint)
            used_callbacks.append(sparse_model_weights_checkpoint)
        elif self.params.loss_func == "categorical_crossentropy":
            categorical_model_checkpoint, categorical_model_weights_checkpoint, categorical_early_stopping = get_catgorical_callbacks(self.params)
            if self.params.early_stopping:
                used_callbacks.append(categorical_early_stopping)
            used_callbacks.append(categorical_model_checkpoint)
            used_callbacks.append(categorical_model_weights_checkpoint)

        # Configure optimizer (use Nadam or Adam and 'binary_crossentropy' or jaccard_coef_loss)
        if self.params.optimizer == 'Adam':
            optimizer = Adam(learning_rate=self.params.learning_rate, decay=self.params.decay, amsgrad=True)
        elif self.params.optimizer == 'AdamW':
            optimizer = AdamW(learning_rate=self.params.learning_rate, weight_decay=self.params.decay, amsgrad=True)
        
        if self.params.loss_func == 'binary_crossentropy':
            self.model.compile(optimizer=optimizer,
                                loss='binary_crossentropy',
                                metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                        jaccard_coef_thresholded, 'accuracy'])
        # SIS: Multi Class Prediction loss func
        elif self.params.loss_func == 'sparse_categorical_crossentropy':
            print("Compiling with Sparse Categorical Crossentropy")
            sparse_cat_loss = keras.losses.SparseCategoricalCrossentropy(ignore_class=self.params.dataset_fill_cls) # ignore fill class 
            self.model.compile(optimizer=optimizer,
                            loss=sparse_cat_loss,
                            metrics=[keras.metrics.SparseCategoricalCrossentropy(ignore_class=self.params.dataset_fill_cls),
                                      keras.metrics.SparseCategoricalAccuracy()]) 
        elif self.params.loss_func == 'categorical_crossentropy':
            print("Compiling with Categorical Crossentropy")
            self.model.compile(optimizer=optimizer,
                            loss=keras.losses.CategoricalCrossentropy(),
                            metrics=[keras.metrics.CategoricalCrossentropy(), jaccard_coef_loss, jaccard_coef,
                                    jaccard_coef_thresholded, keras.metrics.CategoricalAccuracy()]) 

        # save params if compilation succeeds
        self._save_params()

        # Create generators
        image_generator = ImageSequence(self.params, shuffle=True, seed=self.seed, augment_data=self.params.affine_transformation)
        val_generator = ImageSequence(self.params, shuffle=True, seed=self.seed, augment_data=self.params.affine_transformation,
                                      validation_generator=True)

        # Do the training
        print('------------------------------------------')
        print('Start training:')
        history = self.model.fit(image_generator,
                        epochs=self.params.epochs,
                        steps_per_epoch=self.params.steps_per_epoch,
                        verbose=1,
                        workers=8, # 4
                        max_queue_size=12,
                        use_multiprocessing=False,
                        shuffle=self.params.shuffle, # False
                        callbacks=used_callbacks,
                        validation_data=val_generator,
                        validation_steps=None)

        self._save_history(history.history)
    
    def _save_history(self, history):
        np.save(f"{self.params.project_path}reports/Unet/histories/{self.params.modelID}_history.npy", history)

    def load_history(self):
        return np.load(f"{self.params.project_path}reports/Unet/histories/{self.params.modelID}_history.npy", allow_pickle=True).item()

    def _save_params(self , postfix=".txt", delimiter=";"):
        with open(self.params.project_path + 'models/Unet/' + self.params.modelID + '_params' + postfix, 'w') as f:
            f.write(self.params.as_string(delimiter=delimiter))
        # TODO: Jsonify this (json_dumps...)
        #with open(self.params.project_path + 'models/Unet/' + get_model_name(self.params) + '_params.json', 'w') as f:
        #    json.dump(str(self.params.__dict__), f, indent=4)
            #f.write(str(self.params.__dict__))

    def _load_params(self, postfix=".txt", delimiter=";"):
        txt = ""
        with open(self.params.project_path + 'models/Unet/' + self.params.modelID + '_params' + postfix, 'r') as f:
            txt = f.read().replace('\n', '')
            print(txt)

        loaded_params = HParams()
        loaded_params.parse(txt, delimiter=delimiter)

        # self.params = loaded_params
        return loaded_params


    def predict(self, img):
        # Predict batches of patches
        patches = np.shape(img)[0]  # Total number of patches
        patch_batch_size = 128

        # Do the prediction
        predicted = np.zeros((patches, self.params.patch_size - self.params.overlap, self.params.patch_size - self.params.overlap, self.n_cls))
        for i in range(0, patches, patch_batch_size):
            predicted[i:i + patch_batch_size, :, :, :] = self.model.predict(img[i:i + patch_batch_size, :, :, :])

        return predicted
