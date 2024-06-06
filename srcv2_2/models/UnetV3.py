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

        # @ https://github.com/rklen/statistical_tests_for_CNNs/blob/main/stm.ipynb
        
        clip_pixels = np.int32(self.params.overlap / 2)  #(self.params.overlap) / 2  # Only used for input in Cropping2D function on next line
        
        #model128 =self._load_model_128()

        model256 = self._load_model_256()

        # model256_2 = self._load_model_256_2()
        
        model = model256
        return model
    
    def _load_model_128(self):
        return keras.models.Sequential([
            keras.layers.Conv2D(16, 3, activation='relu', input_shape=(self.params.patch_size, self.params.patch_size, self.n_bands)),
            keras.layers.Conv2D(16, 3, activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.MaxPooling2D(strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.n_cls, activation='softmax')
        ] 
        )

    def _load_model_256_2(self):
        """
        heavily modified and parametrized upscale of Rainio 2024
        """
        return keras.models.Sequential([
            keras.layers.Conv2D(16, 3, activation=self.params.activation_func, input_shape=(self.params.patch_size, self.params.patch_size, self.n_bands),
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.Conv2D(16, 3, activation=self.params.activation_func,
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(32, 3, activation=self.params.activation_func,
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.Conv2D(32, 3, activation=self.params.activation_func,
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(64, 3, activation=self.params.activation_func,
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.Conv2D(64, 3, activation=self.params.activation_func,
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(128, 3, activation=self.params.activation_func,
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.Conv2D(128, 3, activation=self.params.activation_func,
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(256, 3, activation=self.params.activation_func,
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.Conv2D(256, 3, activation=self.params.activation_func,
                                kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization,
                                padding='same'),
            keras.layers.MaxPooling2D(strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation=self.params.activation_func,
                               kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization),
            keras.layers.Dense(128, activation=self.params.activation_func,
                               kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization),
            keras.layers.Dense(64, activation=self.params.activation_func,
                               kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization),
            keras.layers.Dense(32, activation=self.params.activation_func,
                               kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization),
            #keras.layers.Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels))),
            keras.layers.Dense(self.n_cls, activation=self.params.last_layer_activation_func,
                               kernel_regularizer=regularizers.l2(self.params.L2reg),
                                kernel_initializer=self.params.initialization)
        ] 
        )
    
    def _load_model_256(self):
        """
        slightly modified multi class model from Rainio 2024
        scaled up to 256^2 input
        for 4 (sparse) classes
        @ 7 spectral bands depth
        """
        l2reg = 1e-4
        return keras.models.Sequential([
            keras.layers.Conv2D(16, 3, activation='relu', input_shape=(256, 256, 7),
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.Conv2D(16, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(32, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.Conv2D(32, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(64, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.Conv2D(64, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(128, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.Conv2D(128, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(256, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.Conv2D(256, 3, activation='relu',
                                kernel_regularizer=regularizers.l2(l2reg),
                                kernel_initializer='he_normal',
                                padding='same'),
            keras.layers.MaxPooling2D(strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            #keras.layers.Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels))),
            keras.layers.Dense(4, activation='softmax')
        ] 
        )

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
        csv_logger, model_checkpoint, model_checkpoint_saving, reduce_lr, tensorboard, early_stopping, cyclical_lr_scheduler, factorial_cyclical_lr_scheduler, round_cyclical_learning_rate_scheduler = get_callbacks(self.params)
        used_callbacks = [csv_logger,  tensorboard]
        
        if self.params.reduce_lr:
            used_callbacks.append(reduce_lr)
        if self.params.use_cyclical_lr_scheduler:
            used_callbacks.append(cyclical_lr_scheduler)
        elif self.params.use_factorial_cyclical_lr_scheduler:
            used_callbacks.append(factorial_cyclical_lr_scheduler)
        elif self.params.use_round_cyclical_lr_scheduler:
            used_callbacks.append(round_cyclical_learning_rate_scheduler)
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
