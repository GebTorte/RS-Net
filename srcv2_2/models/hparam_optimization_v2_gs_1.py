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

CLS=['shadow', 'clear', 'thin', 'cloud'] # 'fill' has to be included for categorical? No., so model non-class option for fill pixel and thus wont learn bad habits.
SATELLITE = "Landsat8"
TRAIN_DATASET = "Biome_gt"
TEST_DATASET= TRAIN_DATASET


activation_functions = ['relu'] 
loss_functions = ['sparse_categorical_crossentropy'] # , 'categorical_crossentropy'] # ['binary_crossentropy']
initializers = ['glorot_normal']
learning_rates = [1e-7] # , 1e-7, 1e-8]
img_enhance_funcs = ["enhance_contrast"]
norm_thresholds = [2**16 - 1] # 16-bit-int and max_value of 
use_batch_norm = [True]
batch_norm_momentums = [0.8] 
l2regs = [1e-5, 1e-6]
dropouts = list(reversed([0.25]))  # dropout (0.2, 0.5) !!!
dropout_on_last_layer_only=[False] # True,
decays = list(reversed([1e-3, 1e-5]))
reduce_lrs = [True] # True
early_stoppings = [True] # , True
band_combinations = [[1, 2, 3, 4, 5, 6, 7]] # [[1, 2, 3, 4, 5, 6, 7, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 9], [2, 3, 4, 5], [2, 3, 4], [3]]
epochs = [12]# [3, 10, 20, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200, 200]  # Only used to run random search for longer
collapse_cls = False
overlaps = [40] # has to be of even , 40
train_overlaps = [60] # probably has no effect other than upping training time. Might cause problems with clip_pixels = overlap / 2, if overlap != train_overlap

interpreter = "/home/mxh/anaconda3/envs/tf2+gpu_v2/bin/python3"
script = "/home/mxh/RS-Net/SentinelSemanticSegmentation_v2.py"
# Train the models

for learning_rate in learning_rates:
    for reduce_lr in reduce_lrs:                  
        for l2reg in l2regs:
            for dropout in dropouts:        
                for d_bool in dropout_on_last_layer_only:
                    for decay in decays:
                        for use_bn in use_batch_norm:
                            batch_norm_momentums_dummy = batch_norm_momentums if use_bn else [0.7]
                            for bn_momentum in batch_norm_momentums_dummy:
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
                                        random=False,
                                        shuffle=True,
                                        optimizer='AdamW',
                                        loss_func="sparse_categorical_crossentropy",
                                        learning_rate=learning_rate, # next 1e-6
                                        reduce_lr=reduce_lr, # True maybe?, as it monitors val_loss aswell
                                        plateau_patience=9, #12 # in epochs
                                        use_cyclical_lr_scheduler=False,
                                        use_factorial_cyclical_lr_scheduler=False,
                                        use_round_cyclical_lr_scheduler=False,
                                        early_stopping=False,
                                        early_patience=35, # maybe up this to ~= epochs/2
                                        replace_fill_values = True,
                                        dataset_fill_cls=4, # if set to any number, fill values will be replaced with it and will be ignored by loss calculation. If set to None, fill values will be replaced by most probable cls and not ignored by sparse categorical crossentropy.
                                        L2reg=l2reg, # 1e-4 # next 7e-7 or 3e-6
                                        dropout=dropout, # 0.05, # this a tiny bit maybe? # or not?
                                        dropout_on_last_layer_only=d_bool, # if using dropout, definitely test both
                                        decay=decay,  # 0.2, #this a bit, to slow down learning through Adam, ~ 1e-5 or so
                                        bands=[1, 2, 3, 4, 5, 6, 7],
                                        epochs=epoch_no, # training goes well, maybe just reduce epochs a bit, so less overfitting?
                                        norm_method="enhance_contrast", #"enhance_contrast"
                                        use_batch_norm=use_bn,
                                        batch_norm_momentum=bn_momentum, # increase for stability?
                                        initialization="glorot_normal", # he_normal again next # glorot_normal, try he_normal for (r)elu perhaps?
                                        last_layer_activation_func='softmax', # 'softmax'
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
                                        #"--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa    
                                        "--train",

                                        #"--dev_dataset",
                                        #"--test", # works now, but takes a loong time. # needed for writing csv output.
                                        
                                        "--save_output", # every model has like 3G output
                                        
                                        "--satellite", str(SATELLITE), 

                                        "--params="+params.as_string()])



