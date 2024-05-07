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

activation_functions = ['elu', "leaky_relu",] # 'relu'
leaky_alphas = [0.05, 0.1]
loss_functions = ['sparse_categorical_crossentropy'] # , 'categorical_crossentropy'] # ['binary_crossentropy']
initializers = ['glorot_normal', "he_normal"]
learning_rates = [1e-6, 1e-7, 1e-8]
img_enhance_funcs = [None, "enhance_contrast"]
use_batch_norm = [True, False]
batch_norm_momentums = [0.1, 0.5, 0.7, 0.95]
l2regs = [1e-4, 1e-6, 1e-8]
dropouts = list(reversed([0, 1e-5, 1e-2, 0.2]))
dropout_on_last_layer_only=[False, True] # True,
decays = list(reversed([0, 1e-2, 0.2]))
reduce_lr = [False,True]
early_stoppings = [False, True]
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
epochs = [1, 3, 5, 8, 12]# [3, 10, 20, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200, 200]  # Only used to run random search for longer
last_layer_activation_func = 'softmax'
satellite = "Landsat8"
gt_cls_list=[['fill', 'shadow', 'clear', 'thin', 'cloud'], ['shadow', 'clear', 'thin', 'cloud'], ['clear', 'cloud']] #['fill','shadow', 'clear', 'thin', 'cloud'],  # [['clear', 'cloud', 'shadow', 'snow', 'water']] # [['clear', 'cloud', 'thin', 'shadow']] 
fmask_cls_list = [['clear', 'cloud', 'shadow', 'snow', 'water']] # ['clear', 'cloud']
train_datasets = ["Biome_gt"] #  "Biome_fmask", 
# ['clear', 'shadow', 'thin', 'cloud'] # thin only in Biome_gt
# ['clear', 'cloud', 'shadow', 'snow', 'water']  # this is only possible on BIOME_fmask
collapse_cls = False
overlaps = [10, 40] # has to be of even , 40
# train_overlaps = [10, 40]

# possibly trying without NORMALIZATION 
# normalized_dataset = False ";normalized_dataset=" + str(normalized_dataset) + \ 
#"--normalize_dataset" if normalized_dataset else ""

interpreter = "/home/mxh/anaconda3/envs/tf2+gpu_v2/bin/python3"
script = "/home/mxh/RS-Net/SentinelSemanticSegmentation_v2.py"
# Train the models
for train_dataset in train_datasets:
    cls_list = gt_cls_list if train_dataset == "Biome_gt" else fmask_cls_list
    ## for train_overlap in train_overlaps:
    for overlap in overlaps:
        params = f"--params=\"train_dataset={train_dataset};overlap={overlap};satellite={satellite}\""  # ;overlap_train_set={overlap}
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
                                        for bn_momentum in batch_norm_momentums:
                                            for epoch_no in epochs:                                
                                                for bands in band_combinations:
                                                    for initializer in initializers:
                                                        for img_enhancer in img_enhance_funcs:
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
                                                                        ";norm_method=" + str(img_enhancer) + \
                                                                        ";use_batch_norm=" + str(use_bn) + \
                                                                        ";batch_norm_momentum=" + str(bn_momentum) + \
                                                                        ";dropout_on_last_layer_only=" + str(d_bool) + \
                                                                        ";initialization=" + str(initializer) + \
                                                                        ";last_layer_activation_func=" + str(last_layer_activation_func) + \
                                                                        ";satellite=" + str(satellite) + \
                                                                        ";str_cls=" + str(cls) + \
                                                                        ";int_cls=" + str(int_cls) + \
                                                                        ";cls=" + str(int_cls) + \
                                                                        ";train_dataset=" + str(train_dataset) + \
                                                                        ";collapse_cls=" + str(collapse_cls) + \
                                                                        ";overlap=" + str(overlap) + \
                                                                        "\""
                                                                        
                                                                        #";leaky_alpha=" + str(leaky_alpha) + \
                                                                        #";cls=" + str(int_cls) + \
                                                                        
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
                                                                print('batch_norm_momentum: ' + str(bn_momentum))
                                                                print('img enhancer/norm method: ' + str(img_enhancer))
                                                                print('dropout_on_last_layer_only: ' + str(d_bool))
                                                                print('cls: ', str(""+c for c in cls))
                                                                print('---------------------------------------------------')
                                                                subprocess.check_call([interpreter,
                                                                                    script,
                                                                                    #"--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa    
                                                                                    "--train",

                                                                                    #"--dev_dataset",
                                                                                    "--test", # works now, but takes a loong time. # needed for writing csv output.
                                                                                    "--save_output",
                                                                                    "--satellite", str(satellite), 
                                                                                    params])



