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
##from utils import get_model_name, get_cls
from srcv2_2.models.model_utils import jaccard_coef, jaccard_coef_thresholded, jaccard_coef_loss, swish, get_callbacks, get_sparse_catgorical_callbacks, get_catgorical_callbacks,ImageSequence #, jaccard_coef_loss_sparse_categorical
#from models.model_utils import jaccard_coef, jaccard_coef_thresholded, jaccard_coef_loss, swish, get_callbacks, get_sparse_catgorical_callbacks, get_catgorical_callbacks,ImageSequence #, jaccard_coef_loss_sparse_categorical
from srcv2_2.models.params import HParams
#from models.params import HParams

from keras.models import Model
from keras.layers import *
import keras
from tensorflow.keras.optimizers import *

from keras import backend as K

smooth = 0.0000001


class UnetV4_CXN(object):
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


        model = self.model_arch_256(input_rows=self.params.patch_size,
                                input_cols=self.params.patch_size, 
                                num_of_channels=self.n_bands, 
                                num_of_classes=self.n_cls, 
                                conv2d_kernel_shape=(7,7),
                                initialization=self.params.initialization)
        #model.compile(optimizer = Adam(lr = 1e-4), loss = self.params., metrics = [self.jacc_coef,'accuracy'])

        model.build(tf.TensorShape(dims=[self.params.batch_size, self.params.patch_size, self.params.patch_size, self.n_bands]))

        model.summary(print_fn=UnetV4_CXN.print_summary)

        print("Len model layers:", len(model.layers))

        return model
    
    @staticmethod
    def print_summary(s):
        with open("UnetV4_CXN_summary.html", "a") as f:
            f.write(s + "\n")

    @staticmethod
    def aspp(x,out_shape, L2reg, initialization):
        b0=SeparableConv2D(256,(1,1),padding="same",use_bias=False, depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(x)
        b0=BatchNormalization()(b0)
        b0=Activation("relu")(b0)

        #b5=DepthwiseConv2D((3,3),dilation_rate=(3,3),padding="same",use_bias=False)(x)
        #b5=BatchNormalization()(b5)
        #b5=Activation("relu")(b5)
        #b5=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b5)
        #b5=BatchNormalization()(b5)
        #b5=Activation("relu")(b5)
        
        b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False, depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(x)
        b1=BatchNormalization()(b1)
        b1=Activation("relu")(b1)
        b1=SeparableConv2D(256,(1,1),padding="same",use_bias=False, depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(b1)
        b1=BatchNormalization()(b1)
        b1=Activation("relu")(b1)
        
        b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False, depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(x)
        b2=BatchNormalization()(b2)
        b2=Activation("relu")(b2)
        b2=SeparableConv2D(256,(1,1),padding="same",use_bias=False, depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(b2)
        b2=BatchNormalization()(b2)
        b2=Activation("relu")(b2)	
        
        b3=DepthwiseConv2D((3,3),dilation_rate=(18,18),padding="same",use_bias=False, depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(x)
        b3=BatchNormalization()(b3)
        b3=Activation("relu")(b3)
        b3=SeparableConv2D(256,(1,1),padding="same",use_bias=False, depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(b3)
        b3=BatchNormalization()(b3)
        b3=Activation("relu")(b3)
        
        b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)
        b4=SeparableConv2D(256,(1,1),padding="same",use_bias=False, depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(b4)
        b4=BatchNormalization()(b4)
        b4=Activation("relu")(b4)
        b4=UpSampling2D((out_shape,out_shape), interpolation='bilinear')(b4)
        x=Concatenate()([b4,b0,b1,b2,b3])
        return x




    @staticmethod
    def jacc_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

    @staticmethod
    def bn_relu(input_tensor):
        """It adds a Batch_normalization layer before a Relu
        """
        input_tensor = BatchNormalization(axis=3)(input_tensor)
        return Activation("relu")(input_tensor)


    @staticmethod
    def contr_arm(input_tensor, filters, kernel_size, L2reg, initialization):
        """It adds a feedforward signal to the output of two following conv layers in contracting path
        TO DO: remove keras.layers.add and replace it with add only
        """

        x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(input_tensor)
        x = UnetV4_CXN.bn_relu(x)

        x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(x)
        x = UnetV4_CXN.bn_relu(x)

        filters_b = filters // 2
        kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

        x1 = SeparableConv2D(filters_b, kernel_size_b, padding='same', depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(input_tensor)
        x1 = UnetV4_CXN.bn_relu(x1)

        x1 = concatenate([input_tensor, x1], axis=3)
        x = keras.layers.add([x, x1])
        x = Activation("relu")(x)
        return x
    



    @staticmethod
    def imprv_contr_arm(input_tensor, filters, kernel_size, L2reg, initialization):
        """It adds a feedforward signal to the output of two following conv layers in contracting path
        """

        x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(input_tensor)
        x = UnetV4_CXN.bn_relu(x)

        x0 = SeparableConv2D(filters, kernel_size, padding='same', depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(x)
        x0 = UnetV4_CXN.bn_relu(x0)

        x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(x0)
        x = UnetV4_CXN.bn_relu(x)

        filters_b = filters // 2
        kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

        x1 = SeparableConv2D(filters_b, kernel_size_b, padding='same', depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(input_tensor)
        x1 = UnetV4_CXN.bn_relu(x1)

        x1 = concatenate([input_tensor, x1], axis=3)

        x2 = SeparableConv2D(filters, kernel_size_b, padding='same', depthwise_regularizer=regularizers.l2(L2reg), depthwise_initializer=initialization)(x0)
        x2 = UnetV4_CXN.bn_relu(x2)

        x = keras.layers.add([x, x1, x2])
        x = Activation("relu")(x)
        return x


    @staticmethod
    def bridge(input_tensor, filters, kernel_size, L2reg, initialization):
        """It is exactly like the identity_block plus a dropout layer. This block only uses in the valley of the UNet
        """

        x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_regularizer=regularizers.l2(L2reg), bias_regularizer=regularizers.l2(L2reg),
                             depthwise_initializer=initialization)(input_tensor)
        x = UnetV4_CXN.bn_relu(x)

        x = SeparableConv2D(filters, kernel_size, padding='same', depthwise_regularizer=regularizers.l2(L2reg), bias_regularizer=regularizers.l2(L2reg),
                             depthwise_initializer=initialization)(x)
        x = Dropout(.15)(x)
        x = UnetV4_CXN.bn_relu(x)

        filters_b = filters // 2
        kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

        x1 =SeparableConv2D(filters_b, kernel_size_b, padding='same', depthwise_regularizer=regularizers.l2(L2reg), bias_regularizer=regularizers.l2(L2reg),
                             depthwise_initializer=initialization)(input_tensor)
        x1 = UnetV4_CXN.bn_relu(x1)

        x1 = concatenate([input_tensor, x1], axis=3)
        x = keras.layers.add([x, x1])
        x = Activation("relu")(x)
        return x


    @staticmethod
    def conv_block_exp_path(input_tensor, filters, kernel_size, l2reg, initialization):
        """It Is only the convolution part inside each expanding path's block
        """

        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(l2reg),
                        kernel_initializer=initialization)(input_tensor)
        x = UnetV4_CXN.bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(l2reg),
                        kernel_initializer=initialization)(x)
        x = UnetV4_CXN.bn_relu(x)
        return x


    @staticmethod
    def conv_block_exp_path3(input_tensor, filters, kernel_size, l2reg, initialization):
        """It Is only the convolution part inside each expanding path's block
        """

        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(l2reg),
                        kernel_initializer=initialization)(input_tensor)
        x = UnetV4_CXN.bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(l2reg),
                        kernel_initializer=initialization)(x)
        x = UnetV4_CXN.bn_relu(x)

        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(l2reg),
                        kernel_initializer=initialization)(x)
        x = UnetV4_CXN.bn_relu(x)
        return x


    @staticmethod
    def add_block_exp_path(input_tensor1, input_tensor2, input_tensor3):
        """It is for adding two feed forwards to the output of the two following conv layers in expanding path
        """

        x = keras.layers.add([input_tensor1, input_tensor2, input_tensor3])
        x = Activation("relu")(x)
        return x


    @staticmethod
    def improve_ff_block4(input_tensor1, input_tensor2 ,input_tensor3, input_tensor4, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        TO DO: shrink all of ff blocks in one function/class
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
            x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = concatenate([x2, input_tensor2], axis=3)
        x2 = MaxPooling2D(pool_size=(4, 4))(x2)

        for ix in range(7):
            if ix == 0:
                x3 = input_tensor3
            x3 = concatenate([x3, input_tensor3], axis=3)
        x3 = MaxPooling2D(pool_size=(8, 8))(x3)

        for ix in range(15):
            if ix == 0:
                x4 = input_tensor4
            x4 = concatenate([x4, input_tensor4], axis=3)
        x4 = MaxPooling2D(pool_size=(16, 16))(x4)

        x = keras.layers.add([x1, x2, x3, x4, pure_ff])
        x = Activation("relu")(x)
        return x


    @staticmethod
    def improve_ff_block3(input_tensor1, input_tensor2, input_tensor3, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = concatenate([x2, input_tensor2], axis=3)
        x2 = MaxPooling2D(pool_size=(4, 4))(x2)

        for ix in range(7):
            if ix == 0:
                x3 = input_tensor3
            x3 = concatenate([x3, input_tensor3], axis=3)
        x3 = MaxPooling2D(pool_size=(8, 8))(x3)

        x = keras.layers.add([x1, x2, x3, pure_ff])
        x = Activation("relu")(x)
        return x


    @staticmethod
    def improve_ff_block2(input_tensor1, input_tensor2, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        for ix in range(3):
            if ix == 0:
                x2 = input_tensor2
            x2 = concatenate([x2, input_tensor2], axis=3)
        x2 = MaxPooling2D(pool_size=(4, 4))(x2)

        x = keras.layers.add([x1, x2, pure_ff])
        x = Activation("relu")(x)
        return x


    @staticmethod
    def improve_ff_block1(input_tensor1, pure_ff):
        """It improves the skip connection by using previous layers feature maps
        """

        for ix in range(1):
            if ix == 0:
                x1 = input_tensor1
            x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

        x = keras.layers.add([x1, pure_ff])
        x = Activation("relu")(x)
        return x
    
    def model_arch_128(self, input_rows=256, input_cols=256, num_of_channels=7, num_of_classes=4, conv2d_kernel_shape=(3,3), initialization="glorot_uniform"):
        inputs = Input((input_rows, input_cols, num_of_channels))
        conv1 = Conv2D(16, conv2d_kernel_shape, activation='relu', padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg),
                        kernel_initializer=initialization)(inputs)

        conv1 = UnetV4_CXN.contr_arm(conv1, 32, conv2d_kernel_shape, self.params.L2reg, initialization)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = UnetV4_CXN.contr_arm(pool1, 64, conv2d_kernel_shape, self.params.L2reg, initialization)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = UnetV4_CXN.imprv_contr_arm(pool2, 128, conv2d_kernel_shape, self.params.L2reg, initialization)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = UnetV4_CXN.bridge(pool3, 256, conv2d_kernel_shape, L2reg=self.params.L2reg, initialization=initialization)
        
        conv6  = UnetV4_CXN.aspp(conv4,input_rows/32, L2reg=self.params.L2reg, initialization=initialization)

        convT9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg), kernel_initializer=initialization)(conv6)
        prevup9 = UnetV4_CXN.improve_ff_block2(input_tensor1=conv2, input_tensor2=conv1, pure_ff=conv3)
        up9 = concatenate([convT9, prevup9], axis=3)
        conv9 = UnetV4_CXN.conv_block_exp_path(input_tensor=up9, filters=128, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv9 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv9, input_tensor2=conv3, input_tensor3=convT9)

        convT10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg), kernel_initializer=initialization)(conv9)
        prevup10 = UnetV4_CXN.improve_ff_block1(input_tensor1=conv1, pure_ff=conv2)
        up10 = concatenate([convT10, prevup10], axis=3)
        conv10 = UnetV4_CXN.conv_block_exp_path(input_tensor=up10, filters=64, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv10 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv10, input_tensor2=conv2, input_tensor3=convT10)

        convT11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg), kernel_initializer=initialization)(conv10)
        up11 = concatenate([convT11, conv1], axis=3)
        conv11 = UnetV4_CXN.conv_block_exp_path(input_tensor=up11, filters=32, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv11 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv11, input_tensor2=conv1, input_tensor3=convT11)

        # -----------------------------------------------------------------------
        # add cropping from RS-Net
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv11)
        # -----------------------------------------------------------------------
        
        conv12 = Conv2D(num_of_classes, (1, 1), activation=self.params.last_layer_activation_func)(crop9)
        #conv12 = Conv2D(num_of_classes, (1, 1), activation='softmax')(conv11) # sigmoid

        return Model(inputs=[inputs], outputs=[conv12])
    
    def model_arch_256(self, input_rows=256, input_cols=256, num_of_channels=7, num_of_classes=4, conv2d_kernel_shape=(3,3), initialization="glorot_uniform"):
        inputs = Input((input_rows, input_cols, num_of_channels))
        conv1 = Conv2D(16, conv2d_kernel_shape, activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.params.L2reg),
                        kernel_initializer=initialization)(inputs)

        conv1 = UnetV4_CXN.contr_arm(conv1, 32, conv2d_kernel_shape, self.params.L2reg, initialization)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = UnetV4_CXN.contr_arm(pool1, 64, conv2d_kernel_shape, self.params.L2reg, initialization)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = UnetV4_CXN.contr_arm(pool2, 128, conv2d_kernel_shape, self.params.L2reg, initialization)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = UnetV4_CXN.imprv_contr_arm(pool3, 256, conv2d_kernel_shape, self.params.L2reg, initialization)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = UnetV4_CXN.bridge(pool4, 512, conv2d_kernel_shape, self.params.L2reg, initialization)
        
        conv6  = UnetV4_CXN.aspp(conv5,input_rows/32, self.params.L2reg, initialization=initialization)

        convT8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg),
                        kernel_initializer=initialization)(conv6)
        prevup8 = UnetV4_CXN.improve_ff_block3(input_tensor1=conv3, input_tensor2=conv2, input_tensor3=conv1, pure_ff=conv4)
        up8 = concatenate([convT8, prevup8], axis=3)
        conv8 = UnetV4_CXN.conv_block_exp_path(input_tensor=up8, filters=256, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv8 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv8, input_tensor2=conv4, input_tensor3=convT8)

        convT9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg),
                        kernel_initializer=initialization)(conv8)
        prevup9 = UnetV4_CXN.improve_ff_block2(input_tensor1=conv2, input_tensor2=conv1, pure_ff=conv3)
        up9 = concatenate([convT9, prevup9], axis=3)
        conv9 = UnetV4_CXN.conv_block_exp_path(input_tensor=up9, filters=128, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv9 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv9, input_tensor2=conv3, input_tensor3=convT9)

        convT10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg),
                        kernel_initializer=initialization)(conv9)
        prevup10 = UnetV4_CXN.improve_ff_block1(input_tensor1=conv1, pure_ff=conv2)
        up10 = concatenate([convT10, prevup10], axis=3)
        conv10 = UnetV4_CXN.conv_block_exp_path(input_tensor=up10, filters=64, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv10 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv10, input_tensor2=conv2, input_tensor3=convT10)

        convT11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg),
                        kernel_initializer=initialization)(conv10)
        up11 = concatenate([convT11, conv1], axis=3)
        conv11 = UnetV4_CXN.conv_block_exp_path(input_tensor=up11, filters=32, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv11 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv11, input_tensor2=conv1, input_tensor3=convT11)

        # -----------------------------------------------------------------------
        # add cropping from RS-Net
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv11)
        # -----------------------------------------------------------------------
        
        conv12 = Conv2D(num_of_classes, (1, 1), activation='softmax')(crop9) # sigmoid

        #conv12 = Conv2D(num_of_classes, (1, 1), activation='softmax')(conv11) # sigmoid

        return Model(inputs=[inputs], outputs=[conv12])


    def model_arch(self, input_rows=256, input_cols=256, num_of_channels=7, num_of_classes=4, conv2d_kernel_shape=(3,3), initialization="glorot_uniform"):
        inputs = Input((input_rows, input_cols, num_of_channels))
        conv1 = Conv2D(16, conv2d_kernel_shape, activation='relu', padding='same')(inputs)

        conv1 = UnetV4_CXN.contr_arm(conv1, 32, conv2d_kernel_shape, self.params.L2reg, initialization=initialization)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = UnetV4_CXN.contr_arm(pool1, 64, conv2d_kernel_shape, self.params.L2reg, initialization=initialization)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = UnetV4_CXN.contr_arm(pool2, 128, conv2d_kernel_shape, self.params.L2reg, initialization=initialization)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = UnetV4_CXN.contr_arm(pool3, 256, conv2d_kernel_shape, self.params.L2reg, initialization=initialization)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = UnetV4_CXN.imprv_contr_arm(pool4, 512, conv2d_kernel_shape, self.params.L2reg, initialization=initialization)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        conv6 = UnetV4_CXN.bridge(pool5, 1024, conv2d_kernel_shape, self.params.L2reg, initialization=initialization)
        
        conv6  = UnetV4_CXN.aspp(conv6,input_rows/32, self.params.L2reg, initialization=initialization)

        convT7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg))(conv6)
        prevup7 = UnetV4_CXN.improve_ff_block4(input_tensor1=conv4, input_tensor2=conv3, input_tensor3=conv2, input_tensor4=conv1, pure_ff=conv5)
        up7 = concatenate([convT7, prevup7], axis=3)
        conv7 = UnetV4_CXN.conv_block_exp_path3(input_tensor=up7, filters=512, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv7 = UnetV4_CXN.add_block_exp_path(conv7, conv5, convT7)

        convT8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg))(conv7)
        prevup8 = UnetV4_CXN.improve_ff_block3(input_tensor1=conv3, input_tensor2=conv2, input_tensor3=conv1, pure_ff=conv4)
        up8 = concatenate([convT8, prevup8], axis=3)
        conv8 = UnetV4_CXN.conv_block_exp_path(input_tensor=up8, filters=256, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv8 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv8, input_tensor2=conv4, input_tensor3=convT8)

        convT9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg))(conv8)
        prevup9 = UnetV4_CXN.improve_ff_block2(input_tensor1=conv2, input_tensor2=conv1, pure_ff=conv3)
        up9 = concatenate([convT9, prevup9], axis=3)
        conv9 = UnetV4_CXN.conv_block_exp_path(input_tensor=up9, filters=128, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv9 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv9, input_tensor2=conv3, input_tensor3=convT9)

        convT10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg))(conv9)
        prevup10 = UnetV4_CXN.improve_ff_block1(input_tensor1=conv1, pure_ff=conv2)
        up10 = concatenate([convT10, prevup10], axis=3)
        conv10 = UnetV4_CXN.conv_block_exp_path(input_tensor=up10, filters=64, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv10 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv10, input_tensor2=conv2, input_tensor3=convT10)

        convT11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(self.params.L2reg))(conv10)
        up11 = concatenate([convT11, conv1], axis=3)
        conv11 = UnetV4_CXN.conv_block_exp_path(input_tensor=up11, filters=32, kernel_size=conv2d_kernel_shape, l2reg=self.params.L2reg, initialization=initialization)
        conv11 = UnetV4_CXN.add_block_exp_path(input_tensor1=conv11, input_tensor2=conv1, input_tensor3=convT11)

        # -----------------------------------------------------------------------
        # add cropping from RS-Net
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv11)
        # -----------------------------------------------------------------------
        
        conv12 = Conv2D(num_of_classes, (1, 1), activation='softmax')(crop9) # sigmoid

        #conv12 = Conv2D(num_of_classes, (1, 1), activation='softmax')(conv11) # sigmoid

        return Model(inputs=[inputs], outputs=[conv12])


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
        csv_logger, model_checkpoint, model_checkpoint_saving, model_checkpoint_saving_loss, reduce_lr_on_plat, tensorboard, early_stopping, round_learning_rate_scheduler, smith_cyclical_callback = get_callbacks(self.params)
        used_callbacks = [csv_logger,  tensorboard]
        
        if self.params.reduce_lr:
            used_callbacks.append(reduce_lr_on_plat)
        if self.params.lr_scheduler:
            #used_callbacks.append(smith_cyclical_callback)
            used_callbacks.append(round_learning_rate_scheduler)
        if self.params.loss_func == "binary_crossentropy":
            if self.params.early_stopping:
                used_callbacks.append(early_stopping)
            used_callbacks.append(model_checkpoint)
            used_callbacks.append(model_checkpoint_saving)
            used_callbacks.append(model_checkpoint_saving_loss)
        elif self.params.loss_func == "sparse_categorical_crossentropy":
            sparse_model_checkpoint, sparse_model_weights_checkpoint,  sparse_early_stopping = get_sparse_catgorical_callbacks(self.params)
            if self.params.early_stopping:
                used_callbacks.append(sparse_early_stopping)
            used_callbacks.append(sparse_model_checkpoint)
            used_callbacks.append(sparse_model_weights_checkpoint)
            used_callbacks.append(model_checkpoint_saving_loss)
        elif self.params.loss_func == "categorical_crossentropy":
            categorical_model_checkpoint, categorical_model_weights_checkpoint, categorical_early_stopping = get_catgorical_callbacks(self.params)
            if self.params.early_stopping:
                used_callbacks.append(categorical_early_stopping)
            used_callbacks.append(categorical_model_checkpoint)
            used_callbacks.append(categorical_model_weights_checkpoint)
            used_callbacks.append(model_checkpoint_saving_loss)

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
