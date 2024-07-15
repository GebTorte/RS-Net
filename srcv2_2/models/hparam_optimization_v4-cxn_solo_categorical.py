####################################################
# IMPORTANT: THIS SCRIPT MUST BE RUN FROM TERMIAL! #
####################################################
import subprocess
import numpy as np
import sys
import os
#sys.path.insert(0, '../')
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import get_cls, get_model_name
from params import HParams, get_params


"""
Landsat 8 Order MODIS bands:

L8 | MODGA09
1  |  x      | COASTAL / AEROSOL
2  |  3      | BLUE
3  |  4      | GREEN
4  |  1      | RED
5  |  2      | NIR
6  |  6      | SWIR1
7  |  7      | SWIR2
8  |  x      | PANCHROMATIC
9  |  5      | CIRRUS
10 |  x      | LWIR1
11 |  x      | LWIR2

Order: 3, 4, 1, 2, 6, 7, 5
"""

MODEL = "U-net-v4-CXN"
CLS=['shadow','clear', 'thin', 'cloud'] # try reorder, might reduce loss? ['clear', 'shadow', 'thin', 'cloud']
SATELLITE = "Landsat8"
TRAIN_DATASET = "Biome_gt"
TEST_DATASET= TRAIN_DATASET

interpreter = "/home/mxh/anaconda3/envs/tf2+gpu_v2/bin/python3"
script = "/home/mxh/RS-Net/SentinelSemanticSegmentation_v2.py"

params = get_params(MODEL, SATELLITE)

# define additional parameters
new_params = HParams(activation_func="leaky_relu", # or elu or leaky relu?
                random=False,
                shuffle=True,
                optimizer='AdamW',
                modelID="CXN-256-bn", # "dummy_240629081953-best-val-loss", #"240515092709-CV1of2",
                modelNick="U-net-v4-cxn-256-kernel77", # "U-net-v4-cxn-128-kernel77"
                loss_func="sparse_categorical_crossentropy",
                learning_rate=1e-4,# 1e-6, 1e-4 # 1e-3??
                base_lr=1e-8,
                cyclical_lr_step_size=2*3349, # 2*3349# 2*165 for full? # number iterations per epoch = number batches in epoch ? # for training on full dataset, double?
                cyclical_lr_mode="exp_range", 
                cyclical_lr_gamma = 0.99994,
                batch_size=20, 
                reduce_lr=False, 
                plateau_patience=1,
                lr_scheduler=True,
                # actually use cyclical lr, to combine low val loss of 1e-2 lr and high acc of 1e-5 to 1e-7 ??
                which_scheduler="smith exp_range", #"cyclical-mod6-exp-1.5",#"cyclical-mod6-exp-2", # "custom_scheduler_epoch-cap1_exp-0.7_epsilon1e-10", #"step-6-13--0.5",# "custom_scheduler_epoch-cap4_exp-0.5_epsilon1e-6", # manually adjust this; only for logging
                early_stopping=True,
                early_patience=100, # maybe up this to ~= epochs/2
                replace_fill_values = True,
                dataset_fill_cls=None, # if set to any number, fill values will be replaced by it and it will be ignored by loss calculation. 
                # If set to None, fill values will be replaced by most probable cls and not ignored by sparse categorical crossentropy.
                affine_transformation = True,
                dropout_on_last_layer_only=True, # if using dropout, definitely test both
                decay=0.99, # 1e-3 # initial lr / nr epochs?
                L2reg=1e-5,
                dropout=0.15, # 0.15
                batch_norm_momentum=0,
                bands=[1, 2, 3, 4, 5, 6, 7],
                epochs=42, # set this to x \times modulator -1 to end on a low lr
                # steps_per_epoch=3,
                norm_method="enhance_contrast", #"enhance_contrast"
                initialization="glorot_normal", #he_normal? @rainio2024
                last_layer_activation_func='softmax', # 'softmax'
                satellite=SATELLITE,
                collapse_cls=False,
                cls=CLS,
                str_cls=CLS,
                int_cls=get_cls(SATELLITE, TRAIN_DATASET, CLS),
                train_dataset=TRAIN_DATASET,
                test_dataset=TEST_DATASET, # atm only train=test implemented
                overlap=40, # 20 # 0
                patch_size=256,
                overlap_train_set=60, #120# 6 converts to 3 in every direction, as in fmask
                norm_threshold=2**16-1, #,2**16-1, # 2**16-1, # might set this lower to the max values that actually occur in L8 sensors
                split_dataset=True,
                save_best_only=True)

params.update(**new_params)


if __name__ == '__main__':

    # do the make_dataset step, if training overlap changed, or on train/test dataset change
    # 60 / 120total overlap atm
    #subprocess.check_call([interpreter,
    #                    script,
    #                    "--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa  
    #                    # and for different overlaps/train_dataset_overlaps which can be interdependent, depending on implementation
    #                    "--satellite", str(SATELLITE),
    #                    "--params="+params.as_string()])
    

    # Params string format must fit with the HParams object
    # See more at https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams

    
    print('-------------STARTING NEW--------------------------')
    print(params.as_string(delimiter="\n"))
    print('---------------------------------------------------')
    subprocess.check_call([interpreter,
                        script,
                        "--model", str(MODEL),
                        #"--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa    
                        #"--load_model",
                        "--train",
                        #"--dev_dataset",
                        #"--test", # works now, but takes a loong time. # needed for writing csv output.
                        
                        "--save_output", # every model has like 3G output
                        
                        "--satellite", str(SATELLITE), #

                        "--params="+params.as_string(skip_keys_list=["test_tiles"])])