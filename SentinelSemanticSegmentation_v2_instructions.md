# Instructions for SentinelSemanticSegmentation_v2.py

## Testing a Model:
> python3 SentinelSemanticSegmentation_v2.py --test --satellite Landsat8 --params modelID=Unet_Landsat8_Unet-MOD09GA_240501173116-CV1of2

### also define the model training params!:
>python3 SentinelSemanticSegmentation_v2.py --test --satellite Landsat8 --params modelID=Unet_Landsat8_Unet-MOD09GA_240501173116-CV1of2,cls="['clear', 'shadow', 'thin', 'cloud']"