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
# from keras.optimizers.legacy import Adam, Nadam
from keras.optimizers import AdamW, Adam
from keras.utils import get_custom_objects  # To use swish activation function
#from tensorflow.keras.utils import multi_gpu_model
from ..utils import get_model_name, get_cls
from srcv2_2.models.model_utils import jaccard_coef, jaccard_coef_thresholded, jaccard_coef_loss, swish, get_callbacks, get_sparse_catgorical_callbacks, get_catgorical_callbacks,ImageSequence #, jaccard_coef_loss_sparse_categorical
from srcv2_2.models.params import HParams


class UnetV2(object):
    def __init__(self, params, model=None):
        # Seed for the random generators
        self.seed = 1

        self.params = params
        self.model = model

        try:
            self.params.update(**self._load_params())
            print("Params have been loaded:", self.params.as_string())
        except:
            print("Params were not loaded.")

        # set these random generator seeds to ensure more comparable output. Otherwise train model multiple times and evaluate means
        if not self.params.random:
            numpy.random.seed(1)
            tf.random.set_seed(self.seed)
            keras.utils.set_random_seed(self.seed) # this will apply to dropout aswell (and lead to overfitting i think). 

        # Find the model you would like
        self.model_name = get_model_name(self.params)

        # Find the number of classes and bands
        if self.params.collapse_cls:
            self.n_cls = 1
        else:
            self.n_cls = np.size(self.params.cls)
        self.n_bands = np.size(self.params.bands)

        # setting integer classes for sparse_categorical_crossentropy
        #try:
        #    if self.params.int_cls:
        #        self.params.str_cls = self.params.cls
        #        self.params.cls = self.params.int_cls
        #except AttributeError:
        #    # assuming cls are in str form
        #    self.params.str_cls = self.params.cls
        #    self.params.cls = get_cls(self.params.satellite, self.params.train_dataset, cls_string=self.params.cls)
        #    #self.params.int_cls = get_cls(self.params.satellite, self.params.train_dataset, cls_string=self.params.cls)}

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
                with tf.device("/GPU:0"):
                    self.model = self.__create_inference__()  # initialize the model
                # try:
                #     self.model.load_weights(self.params.project_path + 'models/Unet/' + self.model_name)
                #     print('Weights loaded from model: ' + self.model_name)
                # except:
                #     print('No weights found')

            else:
                print("ATTENTION: Only running on CPU")
                with tf.device("/cpu:0"):
                    self.model = self.__create_inference__()  # initialize the model on the CPU
                    # try:
                    #     self.model.load_weights(self.params.project_path + 'models/Unet/' + self.model_name)
                    #     print('Weights loaded from model: ' + self.model_name)
                    # except:
                    #     print('No weights found')
            # deprecated -> use tf.distribute.MirroredStrategy().scope() for compiling and training on mulitple gpus instead
            # self.model = multi_gpu_model(self.model, gpus=self.params.num_gpus)  # Make it run on multiple GPUs
        else:
            print(f"Model {get_model_name(self.params)} has been provided in constructor.")
            return

    def __create_inference__(self):
        # Note about BN and dropout: https://stackoverflow.com/questions/46316687/how-to-include-batch-normalization-in-non-sequential-keras-model
        get_custom_objects().update({'swish': Activation(swish)})
        if self.params.activation_func == "leaky_relu":
            activation_func = LeakyReLU(alpha=self.params.leaky_alpha)
        else:
            activation_func = self.params.activation_func

        # conv2d etc kernel initializers default to glorot_uniform
        #if self.params.initialization == "glorot_normal":
        #    kernel_initializer = GlorotNormal(self.seed)
        #elif self.params.initialization == "he_normal":
        #    kernel_initializer = HeNormal(self.seed)

        inputs = Input((self.params.patch_size, self.params.patch_size, self.n_bands))
        # -----------------------------------------------------------------------
        conv1 = Conv2D(32, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(inputs)
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
        up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = Conv2D(256, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up6)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        conv6 = Conv2D(256, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv6)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        # -----------------------------------------------------------------------
        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up7)
        conv7 = Dropout (self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        conv7 = Conv2D(128, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv7)
        conv7 = Dropout (self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        # -----------------------------------------------------------------------
        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(up8)
        conv8 = Dropout (self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        conv8 = Conv2D(64, (3, 3), activation=activation_func, padding='same',
                       kernel_regularizer=regularizers.l2(self.params.L2reg),
                       kernel_initializer=self.params.initialization)(conv8)
        conv8 = Dropout (self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
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
                                      keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseTopKCategoricalAccuracy(k=2)]) 
        elif self.params.loss_func == 'categorical_crossentropy':
            print("Compiling with Categorical Crossentropy")
            #probability = 1/self.n_cls
            #logits = tf.constant([[probability]*self.n_cls] *self.n_cls) # uniform distribution
            #labels = tf.constant([((i-1)*[0] + [1] + (self.n_cls-i) * [0]) for i in range(1, self.n_cls+1)]) # one hot encoded
            #labels = tf.constant([[1, 0, 0], [0, 1, 0]]) # one hot encoded
            #print(logits, labels)
            #cat_loss = keras.losses.CategoricalCrossentropy(from_logits=False)(labels, logits)
            self.model.compile(optimizer=optimizer,
                            loss=keras.losses.CategoricalCrossentropy(),
                            metrics=[keras.metrics.CategoricalCrossentropy(), jaccard_coef_loss, jaccard_coef,
                                    jaccard_coef_thresholded, keras.metrics.CategoricalAccuracy()]) 
                                    # drop keras.metrics.Accuracy()
                                    # 'accuracy' will be converted to CategoricalAccuracy by tf in this case
        elif self.params.loss_func == 'jaccard_coef_loss':
            self.model.compile(optimizer=Adam(lr=self.params.learning_rate, decay=self.params.decay, amsgrad=True),
                                loss=jaccard_coef_loss,
                                metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                        jaccard_coef_thresholded, 'accuracy'])
        elif self.params.optimizer == 'Nadam':
            if self.params.loss_func == 'binary_crossentropy':
                self.model.compile(optimizer=Nadam(lr=self.params.learning_rate),
                                   loss='binary_crossentropy',
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])
            elif self.params.loss_func == 'jaccard_coef_loss':
                self.model.compile(optimizer=Nadam(lr=self.params.learning_rate),
                                   loss=jaccard_coef_loss,
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])

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
                        shuffle=True, # False
                        callbacks=used_callbacks,
                        validation_data=val_generator,
                        validation_steps=None)

        # Save the weights (append the val score in the name)
        # There is a bug with multi_gpu_model (https://github.com/kuza55/keras-extras/issues/3), hence model.layers[-2]
        # self.model.save_weights(self.params.project_path + 'models/Unet/' + get_model_name(self.params))
        # self.model.save(self.params.project_path + 'models/Unet/' + self.model_name + '.keras')
        # self.model.save(self.params.project_path + 'models/Unet/' + get_model_name(self.params) + '.keras')
        # model saving is done by checkpoint now
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



#@keras.saving.register_keras_serializable()
class Unet(object):
    def __init__(self, params):
        # Seed for the random generators
        self.seed = 1
        self.params = params

        # Find the model you would like
        self.model_name = get_model_name(self.params)

        # Find the number of classes and bands
        if self.params.collapse_cls:
            self.n_cls = 1
        else:
            self.n_cls = np.size(self.params.cls)
        self.n_bands = np.size(self.params.bands)

        # Create the model in keras
        if self.params.num_gpus == 1:
            self.model = self.__create_inference__()  # initialize the model
            try:
                self.model.load_weights(self.params.project_path + 'models/Unet/' + self.model_name)
                print('Weights loaded from model: ' + self.model_name)
            except:
                print('No weights found')

        else:
            with tf.device("/cpu:0"):
                self.model = self.__create_inference__()  # initialize the model on the CPU
                try:
                    self.model.load_weights (self.params.project_path + 'models/Unet/' + self.model_name)
                    print('Weights loaded from model: ' + self.model_name)
                except:
                    print('No weights found')
            # deprecated -> use tf.distribute.MirroredStrategy().scope() for compiling and training on mulitple gpus instead
            # self.model = multi_gpu_model(self.model, gpus=self.params.num_gpus)  # Make it run on multiple GPUs

    def __create_inference__(self):
        # Note about BN and dropout: https://stackoverflow.com/questions/46316687/how-to-include-batch-normalization-in-non-sequential-keras-model
        get_custom_objects().update({'swish': Activation(swish)})
        inputs = Input( (self.params.patch_size, self.params.patch_size, self.n_bands))
        # -----------------------------------------------------------------------
        conv1 = Conv2D(32, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(inputs)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        conv1 = Conv2D(32, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(conv1)
        conv1 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv1) if self.params.use_batch_norm else conv1
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # -----------------------------------------------------------------------
        conv2 = Conv2D(64, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(pool1)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        conv2 = Conv2D(64, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(conv2)
        conv2 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv2) if self.params.use_batch_norm else conv2
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # -----------------------------------------------------------------------
        conv3 = Conv2D(128, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(pool2)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        conv3 = Conv2D(128, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(conv3)
        conv3 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv3) if self.params.use_batch_norm else conv3
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # -----------------------------------------------------------------------
        conv4 = Conv2D(256, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(pool3)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        conv4 = Conv2D(256, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(conv4)
        conv4 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv4) if self.params.use_batch_norm else conv4
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        # -----------------------------------------------------------------------
        conv5 = Conv2D(512, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(pool4)
        conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        conv5 = Conv2D(512, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(conv5)
        conv5 = BatchNormalization(momentum=self.params.batch_norm_momentum)(conv5) if self.params.use_batch_norm else conv5
        # -----------------------------------------------------------------------
        up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = Conv2D(256, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(up6)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        conv6 = Conv2D(256, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(conv6)
        conv6 = Dropout (self.params.dropout)(conv6) if not self.params.dropout_on_last_layer_only else conv6
        # -----------------------------------------------------------------------
        up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = Conv2D(128, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(up7)
        conv7 = Dropout (self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        conv7 = Conv2D(128, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(conv7)
        conv7 = Dropout (self.params.dropout)(conv7) if not self.params.dropout_on_last_layer_only else conv7
        # -----------------------------------------------------------------------
        up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = Conv2D(64, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(up8)
        conv8 = Dropout (self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        conv8 = Conv2D(64, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(conv8)
        conv8 = Dropout (self.params.dropout)(conv8) if not self.params.dropout_on_last_layer_only else conv8
        # -----------------------------------------------------------------------
        up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(32, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(up9)
        conv9 = Dropout (self.params.dropout)(conv9) if not self.params.dropout_on_last_layer_only else conv9
        conv9 = Conv2D(32, (3, 3), activation=self.params.activation_func, padding='same',
                       kernel_regularizer=regularizers.l2 (self.params.L2reg))(conv9)
        conv9 = Dropout (self.params.dropout)(conv9)
        # -----------------------------------------------------------------------
        clip_pixels = np.int32 (self.params.overlap / 2)  # Only used for input in Cropping2D function on next line
        crop9 = Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv9)
        # -----------------------------------------------------------------------

        conv10 = Conv2D(self.n_cls, (1, 1), activation='sigmoid')(crop9)
        
        # -----------------------------------------------------------------------
        model = Model(inputs=inputs, outputs=conv10)

        return model

    def get_config(self):
        return {'seed': self.seed, 'params': self.params, 'n_cls': self.n_cls, 'n_bands': self.n_bands, 'model_config': self.model.get_config()}

    def train(self):
        # Define callbacks
        csv_logger, model_checkpoint, reduce_lr, tensorboard, early_stopping = get_callbacks(self.params)
        used_callbacks = [csv_logger, model_checkpoint, tensorboard]
        if self.params.reduce_lr:
            used_callbacks.append(reduce_lr)
        if self.params.early_stopping:
            used_callbacks.append(early_stopping)

        # Configure optimizer (use Nadam or Adam and 'binary_crossentropy' or jaccard_coef_loss)
        if self.params.optimizer == 'Adam':
            if self.params.loss_func == 'binary_crossentropy':
                self.model.compile(optimizer=Adam(lr=self.params.learning_rate, decay=self.params.decay, amsgrad=True),
                                   loss='binary_crossentropy',
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])
            elif self.params.loss_func == 'jaccard_coef_loss':
                self.model.compile(optimizer=Adam(lr=self.params.learning_rate, decay=self.params.decay, amsgrad=True),
                                   loss=jaccard_coef_loss,
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])
        elif self.params.optimizer == 'Nadam':
            if self.params.loss_func == 'binary_crossentropy':
                self.model.compile(optimizer=Nadam(lr=self.params.learning_rate),
                                   loss='binary_crossentropy',
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])
            elif self.params.loss_func == 'jaccard_coef_loss':
                self.model.compile(optimizer=Nadam(lr=self.params.learning_rate),
                                   loss=jaccard_coef_loss,
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])

        # Create generators
        image_generator = ImageSequence(self.params, shuffle=True, seed=self.seed, augment_data=self.params.affine_transformation)
        val_generator = ImageSequence(self.params, shuffle=True, seed=self.seed, augment_data=self.params.affine_transformation,
                                      validation_generator=True)

        # Do the training
        print('------------------------------------------')
        print('Start training:')
        # deprecated method
        #self.model.fit_generator(
        self.model.fit_generator(image_generator,
                        epochs=self.params.epochs,
                        steps_per_epoch=self.params.steps_per_epoch,
                        verbose=1,
                        workers=4,
                        max_queue_size=16,
                        use_multiprocessing=True,
                        shuffle=False,
                        callbacks=used_callbacks,
                        validation_data=val_generator,
                        validation_steps=None)

        # Save the weights (append the val score in the name)
        # There is a bug with multi_gpu_model (https://github.com/kuza55/keras-extras/issues/3), hence model.layers[-2]
        if self.params.num_gpus != 1:
            self.model = self.model.layers[-2]
            self.model.save_weights(self.params.project_path + 'models/Unet/' + self.model_name)
            self.model = multi_gpu_model(self.model, gpus=self.params.num_gpus)  # Make it run on multiple GPUs
        else:
            self.model.save_weights(self.params.project_path + 'models/Unet/' + self.model_name)
            self.model.save(self.params.project_path + 'models/Unet/' + self.model_name + '.keras')

    def predict(self, img):
        # Predict batches of patches
        patches = np.shape(img)[0]  # Total number of patches
        patch_batch_size = 128

        # Do the prediction
        predicted = np.zeros((patches, self.params.patch_size - self.params.overlap, self.params.patch_size - self.params.overlap, self.n_cls))
        for i in range(0, patches, patch_batch_size):
            predicted[i:i + patch_batch_size, :, :, :] = self.model.predict(img[i:i + patch_batch_size, :, :, :])

        return predicted

