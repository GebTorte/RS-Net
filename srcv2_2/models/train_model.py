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
from params import HParams


CLS=['thin', 'cloud'] # ['shadow', 'clear', 'thin', 'cloud'] # 'fill' has to be included for categorical? No., so model non-class option for fill pixel and thus wont learn bad habits.
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

# define additional parameters
params = HParams(activation_func="relu",
                modelID="binary_M1",
                random=False, # False -> reproducible
                shuffle=False,
                split_dataset=False,
                loss_func="binary_crossentropy",
                optimizer="Adam",
                learning_rate=0.97e-4,
                use_cyclical_lr_scheduler=False,
                use_factorial_cyclical_lr_scheduler=False,
                use_round_cyclical_lr_scheduler=False,
                reduce_lr=False, 
                plateau_patience=12, 
                early_patience=100, 
                replace_fill_values = False,
                dataset_fill_cls=None, # will be replaced by most likely cls. For categorical, change this!
                affine_transformation = True,
                L2reg=.99e-4, 
                dropout=0,
                dropout_on_last_layer_only=True,
                decay=0,
                bands=[1, 2, 3, 4, 5, 6, 7],
                epochs=42, 
                norm_method="enhance_contrast", 
                use_batch_norm=True,
                batch_norm_momentum=0.7, 
                initialization="glorot_normal", # glorot_normal for categorical, try he_normal for (r)elu perhaps?
                last_layer_activation_func='sigmoid', # 'softmax'
                satellite=SATELLITE,
                collapse_cls=True,
                cls=CLS,
                str_cls=CLS,
                int_cls=get_cls(SATELLITE, TRAIN_DATASET, CLS),
                train_dataset=TRAIN_DATASET,
                test_dataset=TEST_DATASET, # atm only train=test implemented
                overlap=40, # 20 # 0
                overlap_train_set=60, #120# 6 converts to 3 in every direction, as in fmask
                norm_threshold=2**16-1, # might set this lower to the max values that actually occur in L8 sensors
                save_best_only=False)


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
                        #"--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa    
                        "--train",
                        #"--dev_dataset",
                        "--test", # works now, but takes a loong time. # needed for writing csv output.
                        
                        "--save_output", # every model has like 3G output
                        
                        "--satellite", str(SATELLITE), 

                        "--params="+params.as_string()])
