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

activation_functions = ['elu', 'relu', "leaky_relu"] # , 'relu']
leaky_alphas = [0.05, 0.1]
loss_functions = ['sparse_categorical_crossentropy'] # , 'categorical_crossentropy'] # ['binary_crossentropy']
initializers = ['glorot_normal', "he_normal"]
learning_rates = [1e-5, 1e-6, 1e-7, 1e-8]
use_batch_norm = ['True', 'False']
l2regs = [1e-4, 1e-6, 1e-8]
dropouts = [0, 1e-5, 1e-2]
decays = [0, 1e-6, 1e-4, 1e-2, 0.2]
band_combinations = [[1, 2, 3, 4, 5, 6, 7]] # [[1, 2, 3, 4, 5, 6, 7, 9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 9], [2, 3, 4, 5], [2, 3, 4], [3]]
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
epochs = [3, 5, 8, 12, 20]# [3, 10, 20, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200, 200]  # Only used to run random search for longer
dropout_on_last_layer_only=[False, True] # True,
last_layer_activation_func = 'softmax'
satellite = "Landsat8"
gt_cls_list=[['shadow', 'clear', 'thin', 'cloud']]  # [['clear', 'cloud', 'shadow', 'snow', 'water']] # [['clear', 'cloud', 'thin', 'shadow']] 
fmask_cls_list = [['clear', 'cloud', 'shadow', 'snow', 'water']] # ['clear', 'cloud']
train_datasets = ["Biome_gt"] #  "Biome_fmask", 
# ['clear', 'shadow', 'thin', 'cloud'] # thin only in Biome_gt
# ['clear', 'cloud', 'shadow', 'snow', 'water']  # this is only possible on BIOME_fmask
collapse_cls = False
overlaps = [10, 40] # has to be of even

interpreter = "/home/mxh/anaconda3/envs/tf2+gpu/bin/python"
script = "/home/mxh/RS-Net/SentinelSemanticSegmentation_v2.py"
# Train the models
for train_dataset in train_datasets:
    cls_list = gt_cls_list if train_dataset == "Biome_gt" else fmask_cls_list
    for overlap in overlaps:
        params = f"--params=\"train_dataset={train_dataset};overlap_train_set={overlap};overlap={overlap};satellite={satellite}\""
        subprocess.check_call([interpreter,
                            script,
                            "--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa  
                            # and for different overlaps/train_dataset_overlaps which are interdependent
                            "--satellite", str(satellite), 
                            params])
        for activation_func in activation_functions: 
            for loss_func in loss_functions:
                for learning_rate in learning_rates:                  
                    for l2reg in l2regs:
                        for dropout in dropouts:
                            for decay in decays:
                                for d_bool in dropout_on_last_layer_only:
                                    for use_bn in use_batch_norm:
                                        for epoch_no in epochs:                                
                                            for bands in band_combinations:
                                                for initializer in initializers:
                                                    for cls in cls_list:
                                                        if loss_func == "sparse_categorical_crossentropy":
                                                            int_cls = get_cls(satellite, train_dataset, cls)
                                                            # Hacky way to do to random search by overwriting actual values
                                                            #learning_rate = int(np.random.uniform(10, 100, 1)[0]) * 1e-5  # Cast to int to avoid round() function (issues with floats)
                                                            #l2reg = int(np.random.uniform(1, 1000, 1)[0]) * 1e-5  # 1 to 100 when using ground truth, 1 to 1000 when using fmask
                                                            #dropout = int(np.random.uniform(0, 50, 1)[0]) * 1e-2 * np.random.randint(2, size=1)[0]  # 50% chance for dropout=0
                                                            #epoch_no = int(np.random.lognormal(3, 0.8, 1)[0])


                                                        # Params string format must fit with the HParams object
                                                        # See more at https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams
                                                        params = "--params=\"activation_func=" + activation_func + \
                                                                ";loss_func=" + str(loss_func) + \
                                                                ";learning_rate=" + str(learning_rate) + \
                                                                ";L2reg=" + str(l2reg) + \
                                                                ";dropout=" + str(dropout) + \
                                                                ";decay=" + str(decay) + \
                                                                ";bands=" + str(bands) + \
                                                                ";epochs=" + str(epoch_no) + \
                                                                ";use_batch_norm=" + str(use_bn) + \
                                                                ";dropout_on_last_layer_only=" + str(d_bool) + \
                                                                ";initialization=" + str(initializer) + \
                                                                ";last_layer_activation_func=" + str(last_layer_activation_func) + \
                                                                ";satellite=" + str(satellite) + \
                                                                ";cls=" + str(int_cls) + \
                                                                ";train_dataset=" + str(train_dataset) + \
                                                                ";collapse_cls=" + str(collapse_cls) + \
                                                                ";overlap=" + str(overlap) + \
                                                                    "\""
                                                                
                                                                #";leaky_alpha=" + str(leaky_alpha) + \

                                                        print('-------------STARTING NEW--------------------------')
                                                        print('Activation function: ' + activation_func)
                                                        print('loss_func: ' + str(loss_func))
                                                        print('initializer: ' + str(initializer))
                                                        print('learnings_rate: ' + str(learning_rate))
                                                        print('L2reg: ' + str(l2reg))
                                                        print('dropout: ' + str(dropout))
                                                        print('decay: ' + str(decay))
                                                        print('bands: ' + str(bands))
                                                        print('epochs: ' + str(epoch_no))
                                                        print('use_batch_norm: ' + str(use_bn))
                                                        print('dropout_on_last_layer_only: ' + str(d_bool))
                                                        print('cls: ', str(""+c for c in cls))
                                                        print('---------------------------------------------------')
                                                        subprocess.check_call([interpreter,
                                                                            script,
                                                                            #"--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa    
                                                                            "--train",

                                                                            #"--dev_dataset",
                                                                            "--test", # works now, but takes a loong time. # needed for writing csv output.
                                                                            # "--save_output",
                                                                            "--satellite", str(satellite), 
                                                                            params])



