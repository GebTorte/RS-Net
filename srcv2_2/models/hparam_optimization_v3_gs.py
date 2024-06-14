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

MODEL = "U-net-v3"
CLS=['shadow', 'clear', 'thin', 'cloud'] # 'fill' has to be included for categorical? No., so model non-class option for fill pixel and thus wont learn bad habits.
SATELLITE = "Landsat8"
TRAIN_DATASET = "Biome_gt"
TEST_DATASET= TRAIN_DATASET


activation_functions = ['relu'] 
loss_functions = ['sparse_categorical_crossentropy'] # , 'categorical_crossentropy'] # ['binary_crossentropy']
initializers = ["he_normal"]
learning_rates = [1e-3, 1e-5] # , 1e-7, 1e-8]
img_enhance_funcs = ["enhance_contrast"]
l2regs = [0.1, 1e-2, 1e-4]
decays = [0.1, 1e-2, 1e-4]
dropouts=[0.0, 0.2]
band_combinations = [[1, 2, 3, 4, 5, 6, 7]] # [[1, 2, 3, 4, 5, 6, 7, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 9], [2, 3, 4, 5], [2, 3, 4], [3]]
epochs = [4]# [3, 10, 20, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200, 200]  # Only used to run random search for longer
collapse_cls = False

interpreter = "/home/mxh/anaconda3/envs/tf2+gpu_v2/bin/python3"
script = "/home/mxh/RS-Net/SentinelSemanticSegmentation_v2.py"



if __name__ == "__main__":
    #params = HParams(overlap=40, 
    #                 overlap_train_set=60,
    #                 satellite=SATELLITE,
    #                 train_dataset=TRAIN_DATASET) # Might cause problems with clip_pixels = overlap / 2, if overlap != train_overlap
    
    # do the make_dataset step, if training overlap changed, or on train/test dataset change
    # 60 / 120total overlap atm
    #subprocess.check_call([interpreter,
    #                    script,
    #                    "--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa  
    #                    # and for different overlaps/train_dataset_overlaps which can be interdependent, depending on implementation
    #                    "--satellite", str(SATELLITE),
    #                    "--params="+params.as_string()])

    for learning_rate in learning_rates:
        for initializer in initializers:             
            for l2reg in l2regs:
                for decay in decays:
                    for dropout in dropouts:
                        for epoch_no in epochs:
                            # Hacky way to do to random search by overwriting actual values
                            # learning_rate = int(np.random.uniform(10, 9, 1)[0]) * learning_rate  # Cast to int to avoid round() function (issues with floats)
                            #l2reg = int(np.random.uniform(1, 100, 1)[0]) * l2reg  # 1 to 100 when using ground truth, 1 to 1000 when using fmask
                            #dropout = int(np.random.uniform(0, 50, 1)[0]) * 1e-2 * np.random.randint(2, size=1)[0]  # 50% chance for dropout=0
                            #epoch_no = int(np.random.lognormal(3, 0.8, 1)[0])
                            
                            # Params string format must fit with the HParams object
                            # See more at https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams
                            
                            params = HParams(activation_func="relu", # or elu or leaky relu?
                                modelID="dummy", #"240515092709-CV1of2",
                                modelNick="Unet-v3-1024_16",
                                random=False,
                                shuffle=True,
                                optimizer='AdamW',
                                loss_func="sparse_categorical_crossentropy",
                                learning_rate=learning_rate,
                                reduce_lr=True,
                                plateau_patience=12, 
                                which_scheduler="custom_scheduler_epoch-cap1_exp-1.0_epsilon1e-7",
                                early_stopping=False,
                                early_patience=100, # maybe up this to ~= epochs/2
                                replace_fill_values = True,
                                dataset_fill_cls=4, # if set to any number, fill values will be replaced with it and will be ignored by loss calculation. If set to None, fill values will be replaced by most probable cls and not ignored by sparse categorical crossentropy.
                                L2reg=l2reg,
                                decay=decay, 
                                dropout=dropout,
                                bands=[1, 2, 3, 4, 5, 6, 7],
                                epochs=epoch_no,
                                norm_method="enhance_contrast",
                                initialization=initializer, 
                                last_layer_activation_func='softmax',
                                satellite=SATELLITE,
                                collapse_cls=False,
                                cls=CLS,
                                str_cls=CLS,
                                int_cls=get_cls(SATELLITE, TRAIN_DATASET, CLS),
                                train_dataset=TRAIN_DATASET,
                                test_dataset=TEST_DATASET, # atm only train=test implemented
                                overlap=40, # 20 # 0
                                overlap_train_set=60, #120# 6 converts to 3 in every direction, as in fmask
                                norm_threshold=2**16-1, # might set this lower to the max values that actually occur in L8 sensors
                                split_dataset=True,
                                save_best_only=False)
                            
                            print('-------------STARTING NEW--------------------------')
                            print(params.as_string(delimiter="\n"))
                            print('---------------------------------------------------')
                            subprocess.check_call([interpreter,
                                script,
                                "--model", MODEL,

                                #"--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa    
                                "--train",

                                #"--test", # works now, but takes a loong time. # needed for writing csv output.
                                
                                "--save_output", # every model has like 3G output
                                
                                "--satellite", str(SATELLITE), 

                                "--params="+params.as_string()])



