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

activation_functions = ['elu', "leaky_relu",] # 'relu'
leaky_alphas = [0.05, 0.1, 0.2]
loss_functions = ['sparse_categorical_crossentropy'] # , 'categorical_crossentropy'] # ['binary_crossentropy']
initializers = ['glorot_normal', 'he_normal']
learning_rates = [1e-6, 1e-7, 1e-8]
img_enhance_funcs = [None, "enhance_contrast"]
norm_thresholds = [2**16 -1, 25_000] # 16-bit-int and max_value of 
use_batch_norm = [True, False]
batch_norm_momentums = [0.1, 0.5, 0.7, 0.95]
l2regs = [1e-4, 1e-6, 1e-8]
dropouts = list(reversed([0, 1e-4, 1e-2, 0.1]))
dropout_on_last_layer_only=[False, True] # True,
decays = list(reversed([0, 0.2, 1e-2]))
reduce_lr = [True,False]
early_stoppings = [False, True]
ensemble_learnings =  [False]
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
epochs = [3, 5, 8, 12]# [3, 10, 20, 40, 80, 160, 200, 200, 200, 200, 200, 200, 200, 200]  # Only used to run random search for longer
last_layer_activation_func = 'softmax'
satellite = "Landsat8"
BIOME_gt_cls_list=[['fill','shadow', 'clear', 'thin', 'cloud'], ['shadow', 'clear', 'thin', 'cloud'], ['clear', 'cloud'], ['fill', 'clear', 'cloud']] #['fill','shadow', 'clear', 'thin', 'cloud'],  # [['clear', 'cloud', 'shadow', 'snow', 'water']] # [['clear', 'cloud', 'thin', 'shadow']] 
BIOME_fmask_cls_list = [['clear', 'cloud', 'shadow', 'snow', 'water']] # ['clear', 'cloud']
SPARCS_gt_cls_list = [['shadow', 'snow', 'water', 'cloud', 'clear'], ['clear', 'cloud']]
train_datasets = ["SPARCS_gt","Biome_gt"] #  "Biome_fmask", 
# ['clear', 'shadow', 'thin', 'cloud'] # thin only in Biome_gt
# ['clear', 'cloud', 'shadow', 'snow', 'water']  # this is only possible on BIOME_fmask
collapse_cls = False
overlaps = [10, 40] # has to be of even , 40
train_overlaps = [10, 40]

# possibly trying without NORMALIZATION 
# normalized_dataset = False ";normalized_dataset=" + str(normalized_dataset) + \ 
#"--normalize_dataset" if normalized_dataset else ""

interpreter = "/home/mxh/anaconda3/envs/tf2+gpu_v2/bin/python3"
script = "/home/mxh/RS-Net/SentinelSemanticSegmentation_v2.py"
# Train the models
for train_dataset in train_datasets:
    cls_list = BIOME_gt_cls_list if train_dataset == "Biome_gt" else BIOME_fmask_cls_list
    cls_list = SPARCS_gt_cls_list if train_dataset == "SPARCS_gt" else BIOME_gt_cls_list # dummy
    split_flag = True
    ## for train_overlap in train_overlaps:
    for overlap in overlaps:
        params = f"--params=\"train_dataset={train_dataset};overlap={overlap};satellite={satellite};split_dataset={split_flag}\""  # ;overlap_train_set={overlap}
        params = HParams(train_dataset=train_dataset, overlap=overlap, overlap_train_set=0, satellite=satellite, split_dataset=split_flag)
        subprocess.check_call([interpreter,
                            script,
                            "--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa  
                            # and for different overlaps/train_dataset_overlaps which are interdependent
                            "--satellite", str(satellite),
                            "--params="+params.as_string()])
        for activation_func in activation_functions: 
            leaky_alphas = leaky_alphas if activation_func == "leaky_relu" else [0.1] # dummy val
            for leaky_alpha in leaky_alphas:
                for loss_func in loss_functions:
                    for learning_rate in learning_rates:                  
                        for l2reg in l2regs:
                            for dropout in dropouts:
                                for decay in decays:
                                    for d_bool in dropout_on_last_layer_only:
                                        for use_bn in use_batch_norm:
                                            batch_norm_momentums = batch_norm_momentums if use_bn else [0.7] # dummy val
                                            for bn_momentum in batch_norm_momentums:
                                                for epoch_no in epochs:                                
                                                    for bands in band_combinations:
                                                        for initializer in initializers:
                                                            for early_stop in early_stoppings:
                                                                for use_ensemble_learning in ensemble_learnings:
                                                                    for img_enhancer in img_enhance_funcs:
                                                                        norm_thresholds = norm_thresholds if img_enhancer == "enhance_contrast" else [2**16-1] # dummy val
                                                                        for norm_threshold in norm_thresholds:
                                                                            for cls in cls_list:
                                                                                if loss_func == "sparse_categorical_crossentropy":
                                                                                    int_cls = get_cls(satellite, train_dataset, cls)
                                                                                # Hacky way to do to random search by overwriting actual values
                                                                                #learning_rate = int(np.random.uniform(10, 100, 1)[0]) * 1e-5  # Cast to int to avoid round() function (issues with floats)
                                                                                #l2reg = int(np.random.uniform(1, 100, 1)[0]) * 1e-5  # 1 to 100 when using ground truth, 1 to 1000 when using fmask
                                                                                #dropout = int(np.random.uniform(0, 50, 1)[0]) * 1e-2 * np.random.randint(2, size=1)[0]  # 50% chance for dropout=0
                                                                                #epoch_no = int(np.random.lognormal(3, 0.8, 1)[0])
                                                                                

                                                                                # Params string format must fit with the HParams object
                                                                                # See more at https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams
                                                                                
                                                                                params = HParams(activation_func=activation_func,
                                                                                                leaky_alpha=leaky_alpha,
                                                                                                loss_func=loss_func,
                                                                                                learning_rate=learning_rate,
                                                                                                early_stopping=early_stop,
                                                                                                use_ensemble_learning=use_ensemble_learning,
                                                                                                L2reg=l2reg,
                                                                                                dropout=dropout,
                                                                                                decay=decay,
                                                                                                bands=bands,
                                                                                                epochs=epoch_no,
                                                                                                norm_method=img_enhancer,
                                                                                                use_batch_norm=use_bn,
                                                                                                batch_norm_momentum=bn_momentum,
                                                                                                dropout_on_last_layer_only=d_bool,
                                                                                                initialization=initializer,
                                                                                                last_layer_activation_func=last_layer_activation_func,
                                                                                                satellite=satellite,
                                                                                                cls=cls,
                                                                                                str_cls=cls,
                                                                                                int_cls=int_cls,
                                                                                                train_dataset=train_dataset,
                                                                                                test_dataset="Biome_gt",
                                                                                                collapse_cls=collapse_cls,
                                                                                                overlap=overlap,
                                                                                                overlap_train_set=0,
                                                                                                norm_threshold=norm_threshold,
                                                                                                split_dataset=split_flag)
                                                                                
                                                                                print('-------------STARTING NEW--------------------------')
                                                                                print(params.as_string(delimiter="\n"))
                                                                                print('---------------------------------------------------')
                                                                                subprocess.check_call([interpreter,
                                                                                                    script,
                                                                                                    #"--make_dataset",  # needed if cls definitions changed from fmask to gt or vice versa    
                                                                                                    "--train",

                                                                                                    #"--dev_dataset",
                                                                                                    "--test", # works now, but takes a loong time. # needed for writing csv output.
                                                                                                    # "--save_output", # every model has like 3G output
                                                                                                    "--satellite", str(satellite), 

                                                                                                    "--params="+params.as_string()])



