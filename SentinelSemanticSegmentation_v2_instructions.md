# Instructions for SentinelSemanticSegmentation_v2.py

Params are now semi-colon-separeted!

## Making a dataset
Train_dataset defines the processing step

>python3 SentinelSemanticSegmentation_v2.py --make_dataset --params=train_dataset='Biome_gt' --satellite=Landsat8

## Testing a Model:
> python3 SentinelSemanticSegmentation_v2.py --test --satellite Landsat8 --params modelID=Unet_Landsat8_Unet-MOD09GA_240501173116-CV1of2

### also define the model training params!:
! Params has to be a string in "<params>"
! cls has to be in correct order
>  python3 SentinelSemanticSegmentation_v2.py --test --params="modelID=240503193022-CV2of2;cls=['clear','cloud,'thin','shadow'];learning_rate=1e-5"

> python3 SentinelSemanticSegmentation_v2.py --test  --save_output True --params="modelID=SparseCE-Test-2;cls=[0,64,128,192,255];learning_rate=1e-6;decay=0.2;L2reg=1e-4;epochs=5;loss_func=sparse_categorical_crossentropy"

> python3 SentinelSemanticSegmentation_v2.py --test --params="modelID=SparseCE-Test-2;cls=[0,64,128,192,255];learning_rate=1e-6;decay=0.2;L2reg=1e-4;epochs=5;loss_func=sparse_categorical_crossentropy"

> python3 SentinelSemanticSegmentation_v2.py --test --save_output True --params="modelID=SparseCE-Test-2;cls=[64,128,192,255];learning_rate=1e-6;decay=0.2;L2reg=1e-4;epochs=5;loss_func=sparse_categorical_crossentropy;threshold=0.5;dropout=0.1;norm_threshold=65535;overlap=10"

### debug with cmdline args
240507093023-CV2of2
> --test --save_output --params="modelID=240507093023-CV2of2;cls=['shadow','clear','thin','cloud'];learning_rate=1e-5;decay=0;L2reg=1e-4;epochs=5;loss_func=sparse_categorical_crossentropy;threshold=0.5;dropout=0;norm_threshold=65535;overlap=10;activation_func=leaky_relu"

## Training a model

> python3 SentinelSemanticSegmentation_v2.py --train --test --save_output True --params="modelID=240507093023-CV2of2;cls=[64,128,192,255];learning_rate=1e-6;decay=0.2;L2reg=1e-4;epochs=3;loss_func=sparse_categorical_crossentropy;threshold=0.5;dropout=0.01;norm_threshold=65536;overlap=10"


> python3 SentinelSemanticSegmentation_v2.py --test --save_output True --params="modelID=240505215420-CV2of2;cls=[64,128,192,255];learning_rate=1e-6;decay=0.2;L2reg=1e-4;epochs=3;loss_func=sparse_categorical_crossentropy;threshold=0.5;dropout=0.01;norm_threshold=65536;overlap=10"
