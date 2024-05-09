####################################################
# IMPORTANT: THIS SCRIPT MUST BE RUN FROM TERMIAL! #
####################################################
import subprocess
import numpy as np
import sys
import os
#sys.path.insert(0, '../')
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import get_cls
from params import HParams


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

CLS=['fill', 'shadow', 'clear', 'thin', 'cloud']
SATELLITE = "Landsat8"
TRAIN_DATASET = "Biome_gt"

interpreter = "/home/mxh/anaconda3/envs/tf2+gpu_v2/bin/python3"
script = "/home/mxh/RS-Net/SentinelSemanticSegmentation_v2.py"

params = HParams(activation_func="elu",
                leaky_alpha=0.1,
                loss_func="sparse_categorical_crossentropy",
                learning_rate=1e-7,
                reduce_lr=True, # False
                patience=1,
                L2reg=1e-4,
                dropout=0,
                decay=0.1,
                bands=[1, 2, 3, 4, 5, 6, 7],
                epochs=3,
                norm_method="enhance_contrast",
                use_batch_norm=True,
                batch_norm_momentum=0.7,
                dropout_on_last_layer_only=True,
                initialization="glorot_normal",
                last_layer_activation_func='softmax',
                satellite=SATELLITE,
                cls=CLS,
                str_cls=CLS,
                int_cls=get_cls(SATELLITE, TRAIN_DATASET,  CLS),
                train_dataset=TRAIN_DATASET,
                test_dataset=TRAIN_DATASET, # atm only train=test implemented
                collapse_cls=False,
                overlap=10,
                overlap_train_set=0, # 40
                norm_threshold=2**16-1,
                split_dataset=True)


if __name__ == '__main__':

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
                        "--test", # works now, but takes a loong time. # needed for writing csv output.
                        
                        "--save_output", # every model has like 3G output
                        
                        "--satellite", str(SATELLITE), 

                        "--params="+params.as_string()])



