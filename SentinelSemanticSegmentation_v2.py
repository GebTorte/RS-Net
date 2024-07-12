#!/usr/bin/env python
"""
This file is used to run the project. Set all to true to run full pipeline.

Notes:
- The structure of this file (and the entire project in general) is made with emphasis on flexibility for research
purposes, and the pipelining is done in a python file such that newcomers can easily use and understand the code.

- Remember that relative paths in Python are always relative to the current working directory.
Hence, if you look at the functions in make_dataset.py, the file paths are relative to the path of
this file (SentinelSemanticSegmentation.py)
"""

__author__ = "Jacob Høxbroe Jeppesen"
__email__ = "jhj@eng.au.dk"

import time
import argparse
import datetime
import os
import random
import numpy as np
import tensorflow as tf
from srcv2_2.data.make_dataset import make_numpy_dataset
from srcv2_2.models.params import get_params, HParams
from srcv2_2.models.Unet import Unet, UnetV2, get_model_name
from srcv2_2.models.UnetV3 import UnetV3
from srcv2_2.models.UnetV4_CXN import UnetV4_CXN
from srcv2_2.models.evaluate_model import evaluate_test_set, write_csv_files

# Don't allow tensorflow to reserve all memory available
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)  # set this TensorFlow session as the default session for Keras
#set_session(sess)

# ----------------------------------------------------------------------------------------------------------------------
# Define default pipeline
# ----------------------------------------------------------------------------------------------------------------------
# Create the parser. The formatter_class argument makes sure that default values are shown when --help is called.
parser = argparse.ArgumentParser(description='Pipeline for running the project',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Define which steps should be run automatically when this file is run. When using action='store_true', the argument
# has to be provided to run the step. When using action='store_false', the step will be run when this file is executed.
parser.add_argument('--make_dataset',
                    action='store_true',
                    help='Run the pre-processing step')

parser.add_argument('--train',
                    action='store_true',
                    help='Run the training step')

parser.add_argument('--load_model',
                    action='store_true',
                    help='Try loading a saved model')

parser.add_argument('--hparam_optimization',
                    action='store_true',
                    help='Do hyperparameter optimization')

parser.add_argument('--test',
                    action='store_true',
                    help='Run test step')

parser.add_argument('--save_output',
                    action='store_true',
                    help='Save output of testing')


# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments used in the entire pipeline
# ----------------------------------------------------------------------------------------------------------------------
parser.add_argument('--satellite',
                    type=str,
                    default='Landsat8',
                    help='The satellite used (Sentinel-2 or Landsat8)')

parser.add_argument('--initial_model',
                    type=str,
                    default='sen2cor',
                    help='Which initial is model is wanted for training (sen2cor or fmask)')

# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments for the training
# ----------------------------------------------------------------------------------------------------------------------
parser.add_argument('--model',
                    type=str,
                    default='U-net-v2',
                    help='Model type')

parser.add_argument('--params',
                    type=str,
                    help='Semi-colon separated list of "name=value" pairs.')

parser.add_argument('--dev_dataset',
                    action='store_true',
                    help='Very small dataset to be used while developing the project')

parser.add_argument('--normalize_dataset',
                    action='store_true',
                    help='Run enhance contrast over train/val dataset')

# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments for the visualization
# ----------------------------------------------------------------------------------------------------------------------
parser.add_argument('--dataset',
                    type=str,
                    default='Biome',
                    help='Dataset for evaluating Landsat 8 data')


if __name__ == '__main__':
    # Load the arguments
    args = parser.parse_args()
    store_flag = False
    if args.save_output:
        store_flag = True

    # Store current time to calculate execution time later
    start_time = time.time()

    print("\n---------------------------------------")
    print("Script started")
    print("---------------------------------------\n")

    # Load hyperparameters into the params object containing name-value pairs
    params = get_params(args.model, args.satellite)

    # If any hyperparameters were overwritten in the commandline, parse them into params
    if args.params:
        params.parse(args.params)

    # If you want to use local files (else it uses network drive)
    if args.dev_dataset:
        print("using dev_dataset!")
        params.data_path = "/home/mxh/RS-Net/dev_dataset/"

    # Check to see if a new data set should be processed from the raw data
    # important if new training dataset was chosen in params
    if args.make_dataset:
        print("Processing numpy data set")
        #if args.normalize_dataset:
        #    normalize_flag = True
        #else:
        #    normalize_flag = False   
        make_numpy_dataset(params)

    loaded_model = None
    if args.load_model:
        loaded_model = tf.keras.saving.load_model(params.project_path +f"models/Unet/{params.modelID}.keras")

    # Check to see if a model should be trained
    if args.train:
        print("Training " + args.model + " model")
        #training_history = {}
        #hist_index = 0
        if not params.split_dataset:  # No k-fold cross-validation
            # Load the model
            params.modelID = params.modelID + '_' + datetime.datetime.now().strftime("%y%m%d%H%M%S")[:12]
            if args.model == 'U-net-v2':
                model = UnetV2(params, loaded_model)
                model.train()
            elif args.model == 'U-net-v3':
                model = UnetV3(params, loaded_model)
                model.train()
            elif args.model == 'U-net-v4-CXN':
                model = UnetV4_CXN(params, loaded_model)
                model.train()
            # Run model on test data set / leave for test flag
            # evaluate_test_set(model, params.test_dataset, params.num_gpus, params)
        else:  # With k-fold cross-validation
            # Define number of k-folds
            if 'Biome' in params.train_dataset:
                k_folds = 2  # Biome dataset is made for 2-fold CV
            else:
                k_folds = 5  # SPARCS contains 80 scenes, so split it nicely

                # Create a list of names for the splitting
                sparcs_products = sorted(os.listdir(params.project_path + "data/raw/SPARCS_dataset/l8cloudmasks/sending/"))
                sparcs_products = [f for f in sparcs_products if 'data.tif' in f]
                sparcs_products = [f for f in sparcs_products if 'aux' not in f]

                # Randomize the list of SPARCS products
                seed = 1
                random.seed(seed)
                random.shuffle(sparcs_products)

            # Do the training/testing with k-fold cross-validation
            for k in range(k_folds):
                # Define train and test tiles (note that params.test_tiles[0] are training and .test_tiles[1] are test)
                if 'SPARCS' in params.train_dataset:
                    products_per_fold = int(80/k_folds)
                    # Define products for test
                    params.test_tiles[1] = sparcs_products[k*products_per_fold:(k+1)*products_per_fold]
                    # Define products for train by loading all sparcs products and then removing test products
                    params.test_tiles[0] = sparcs_products
                    for product in params.test_tiles[1]:
                        params.test_tiles[0] = [f for f in params.test_tiles[0] if product not in f]

                elif 'Biome' in params.train_dataset:
                    # Swap train and test set for 2-fold CV
                    temp = params.test_tiles[0]
                    params.test_tiles[0] = params.test_tiles[1]
                    params.test_tiles[1] = temp

                # Train and evaluate
                time_stamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
                params.modelID = time_stamp[0:12] + '-CV' + str(k+1) + 'of' + str(k_folds)  # Used for saving results

                if args.model == "U-net-v2":
                    model = UnetV2(params, loaded_model)
                elif args.model == 'U-net-v3':
                    model = UnetV3(params, loaded_model)
                elif args.model == 'U-net-v4-CXN':
                    model = UnetV4_CXN(params, loaded_model)
                
                print("Training on fold " + str(k + 1) + " of " + str(k_folds))
                model.train()

                # Run model on test data set and save output
                evaluate_test_set(model, params.test_dataset, params.num_gpus, params, save_output=store_flag)

    if args.test:
        if args.model == 'U-net-v2':
            #loaded_model = tf.keras.saving.load_model(f"./models/Unet/{params.modelID}.keras")
            # {get_model_name(params)}
            #loaded_model = tf.keras.saving.load_model(f"../models/Unet/{params.modelID}.keras")
            model = UnetV2(params, model=loaded_model)  # to implement for V2: load model from file
        # out = model.evaluate(return_dict=True)
        if args.model == 'U-net-v3':
            #loaded_model = tf.keras.saving.load_model(f"./models/Unet/{params.modelID}.keras")
            # {get_model_name(params)}
            #loaded_model = tf.keras.saving.load_model(f"../models/Unet/{params.modelID}.keras")
            model = UnetV3(params, model=loaded_model)

        if args.model == 'U-net-v4-CXN':
            #loaded_model = tf.keras.saving.load_model(f"./models/Unet/{params.modelID}.keras")
            # {get_model_name(params)}
            #loaded_model = tf.keras.saving.load_model(f"../models/Unet/{params.modelID}.keras")
            model = UnetV4_CXN(params, model=loaded_model)
    
        print("Saving evaluation image output: ", store_flag)
        evaluate_test_set(model, params.test_dataset, params.num_gpus, params, save_output=store_flag) # todo: implement args.save_output

    # Print execution time
    exec_time = str(time.time() - start_time)
    print("\n---------------------------------------")
    print("Script executed in: " + exec_time + "s")
    print("---------------------------------------")
