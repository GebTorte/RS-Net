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

MODEL = "U-net-v2"
CLS=['shadow', 'clear', 'thin', 'cloud'] # 'fill' has to be included for categorical? No., so model non-class option for fill pixel and thus wont learn bad habits.
SATELLITE = "Landsat8"
TRAIN_DATASET = "Biome_gt"
TEST_DATASET= TRAIN_DATASET

interpreter = "/home/mxh/anaconda3/envs/tf2+gpu_v2/bin/python3"
script = "/home/mxh/RS-Net/SentinelSemanticSegmentation_v2.py"

"""
Jeppesen Biome gt x Biome gt, Bands: All non-thermal bands, params:
(for binary cross entropy)

lr: 0.97e-3
l2reg: 0.99e-3
dropout: 0
epochs: 42 (on 1-fold) ->  84 total
adam-decay: 0?
train_set_overlap: 120px -> give 60px to patch_v2 as it cuts from both sides
"""

params = get_params(MODEL, SATELLITE)

# define additional parameters
new_params = HParams(activation_func="relu", # or elu or leaky relu?
                random=False,
                shuffle=True,
                optimizer='AdamW',
                modelID="dummy", #"240515092709-CV1of2",
                loss_func="sparse_categorical_crossentropy",
                learning_rate=1e-2, # up this and use less regulation # was 7e-6
                batch_size=40, # <-- up this?
                use_cyclical_lr_scheduler=False,
                use_factorial_cyclical_lr_scheduler=False,
                use_round_cyclical_lr_scheduler=True,
                reduce_lr=False, # True maybe?, as it monitors val_loss aswell
                plateau_patience=7, # SET this to 1 next, so we hopefully stay at plateau < 10
                early_stopping=True,
                early_patience=19, # maybe up this to ~= epochs/2
                replace_fill_values = True,
                dataset_fill_cls=None, # if no fill, just set this to None
                affine_transformation = True,
                L2reg=2e-1,# -> best go in range 1e-4, 1e-5?
                dropout=0.2, # --> 0.25, # this a significant bit? # or not?
                dropout_on_last_layer_only=False, # if using dropout, definitely test both
                decay=0.3, # initial lr / nr epochs?
                bands=[1, 2, 3, 4, 5, 6, 7],
                epochs=41, # set this to x \times modulator -1 to end on a low lr
                # steps_per_epoch=3,
                norm_method="enhance_contrast", #"enhance_contrast"
                use_batch_norm=True,
                batch_norm_momentum=0.7, # --> 0.7 # was 0.95 #next 0.7, 0.9 # increase for stability
                initialization="glorot_normal", # is this a typo? normal/uniform
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
                norm_threshold=2**16-1, # 2**16-1, # might set this lower to the max values that actually occur in L8 sensors
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
    

    # Hacky way to do to random search by overwriting actual values
    # learning_rate = int(np.random.uniform(10, 9, 1)[0]) * learning_rate  # Cast to int to avoid round() function (issues with floats)
    #l2reg = int(np.random.uniform(1, 100, 1)[0]) * l2reg  # 1 to 100 when using ground truth, 1 to 1000 when using fmask
    #dropout = int(np.random.uniform(0, 50, 1)[0]) * 1e-2 * np.random.randint(2, size=1)[0]  # 50% chance for dropout=0
    #epoch_no = int(np.random.lognormal(3, 0.8, 1)[0])


    # Params string format must fit with the HParams object
    # See more at https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams

    
    print('-------------STARTING NEW--------------------------')
    print(params.as_string(delimiter="\n"))
    print('---------------------------------------------------')
    subprocess.check_call([interpreter,
                        script,
                        #"--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa    
                        "--train",
                        #"--dev_dataset",
                        #"--test", # works now, but takes a loong time. # needed for writing csv output.
                        
                        "--save_output", # every model has like 3G output
                        
                        "--satellite", str(SATELLITE), #

                        "--params="+params.as_string(skip_keys_list=["test_tiles"])])