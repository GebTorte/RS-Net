import numpy as np
import tifffile as tiff
from osgeo import gdal
from osgeo_utils import gdal_merge
from osgeo.gdalconst import GA_ReadOnly

# SECTION: Images
def normalize_float_x(px, scale=1.0, c_min = 0.0, c_max=1.0):
    return float(scale * (px - c_min) / (c_max - c_min) )

vectorized_normalize_float = np.vectorize(normalize_float_x)

def normalize_int_x(px, scale=255.0, c_min = -200, c_max=16000):
    return int(scale * (px - c_min) / (c_max - c_min) )

vectorized_normalize_int_255 = np.vectorize(normalize_int_x)

def make_tc_corrected_img(path, img_name, refl_percent=0.15):
    # refl_percent = 0.15 # assuming 15 percent reflectance gets to sensor
    # img_name = "MOD09GA.A2023363.h18v04.061.2023365041515"
    img_path = path + img_name

    img_tc = img_path + "_merged_2400_true_color.tiff"
    band4 = img_path+ "_b4_2400.tiff"
    band3 = img_path+ "_b3_2400.tiff"
    band1 = img_path+ "_b1_2400.tiff"

    # get reflectance values from [0,1]
    mx = 16000
    r_band_refl = np.clip(tiff.imread(band1)/mx, 0.0, mx)
    b_band_refl = np.clip(tiff.imread(band3)/mx, 0.0, mx)
    g_band_refl = np.clip(tiff.imread(band4)/mx, 0.0, mx)

    true_color_corrected_unscaled = np.dstack((r_band_refl, g_band_refl, b_band_refl))
    true_color_corrected = np.dstack((r_band_refl, g_band_refl, b_band_refl)) * 1/refl_percent

    return true_color_corrected, true_color_corrected_unscaled

def make_tc_corrected_img_from_bands(r_band, g_band, b_band, refl_percent=0.15, nodata=-9999, mx=16000):
    # refl_percent = 0.15 # assuming 15 percent reflectance gets to sensor

    # get reflectance values from [0,1]
    r_band_refl = np.clip(np.divide(r_band, mx), 0.0, mx)
    b_band_refl = np.clip(np.divide(b_band, mx), 0.0, mx)
    g_band_refl = np.clip(np.divide(g_band, mx), 0.0, mx)

    true_color_corrected_unscaled = np.dstack((r_band_refl, g_band_refl, b_band_refl))
    true_color_corrected = np.dstack((r_band_refl, g_band_refl, b_band_refl)) * 1/refl_percent

    return true_color_corrected, true_color_corrected_unscaled


# SECTION: MOD09GA
def read_mod09ga_tiff_refl_bands(path, filename, bands=(0,3,2)):
    """
    Order (before interpolation) from:
    hdf:  [0,3,2]
    tiff (reordered to landsat 8): [2,1,0]
    """
    outputs=[]
    file_data = gdal.Open(path + filename, GA_ReadOnly).ReadAsArray()
    for b in bands:
        outputs.append(file_data[b,:,:])
    return outputs

def merge_mod09ga_refl_bands_to_true_color(path, filename, size=2400, extension=".hdf"):
    """
    filename without .hdf file extension!
    """
    f= gdal.Open(path+filename+extension)
    subsets = f.GetSubDatasets()
    driver = subsets[11][0][:-3] # get common driver command
    # Reorder bands to fit the L8 order. The model is trained on this order.
    L8_order = [3, 4, 1] # [2, 3, 0, 1, 5, 6, 4] # landsat8 order of modis (modga09) bands
    # remember to +1 every band as MODIS starts its bands at index 1
    for i in L8_order:  # translate bands 1-7 to geotiff
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

        Order: 3, 4, 1
        """
        driveri = driver + str(i) +'_1'
        band = gdal.Open(driveri)
        gdal.Translate(destName=f'{path+filename}_b{i}_{size}.tiff', srcDS=band, format="GTiff", width=size, height=size)
    
    #gdal.Warp(destNameOrDestDS=path+filename+'_warped.tif', srcDSOrSrcDSTab=[f'{path+filename}_b{i}.tif' for i in range(1,7+1)])
    
    # merge_params = ['', '-separate']+[f'{path+filename}_b{i}.tif' for i in range(1,7+1)]+['-o', path+filename+'_merged.tif']
    merge_params = ['', '-separate']+[f'{path+filename}_b{i}_{size}.tiff' for i in L8_order]+['-o', path+filename+f'_merged_{size}_true_color.tiff']
    gdal_merge.main(merge_params)

def merge_MOD09GA_TIFF_refl_bands_to_true_color(path, filename, size=2400, extension=".tif"):
    """
    filename without .tif file extension!
    """
    f= gdal.Open(path+filename+extension)
    # Reorder bands to fit the L8 order. The model is trained on this order.
    L8_order = [3, 4, 1] # [2, 3, 0, 1, 5, 6, 4] # landsat8 order of modis (modga09) bands
    # remember to +1 every band as MODIS starts its bands at index 1
    for i in L8_order:  # translate bands 1-7 to geotiff
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

        Order: 3, 4, 1
        """
        driveri = path + filename + extension
        band = gdal.Open(driveri)
        gdal.Translate(destName=f'{path+filename}_b{i}_{size}.tiff', srcDS=band, format="GTiff", width=size, height=size)
    
    #gdal.Warp(destNameOrDestDS=path+filename+'_warped.tif', srcDSOrSrcDSTab=[f'{path+filename}_b{i}.tif' for i in range(1,7+1)])
    
    # merge_params = ['', '-separate']+[f'{path+filename}_b{i}.tif' for i in range(1,7+1)]+['-o', path+filename+'_merged.tif']
    merge_params = ['', '-separate']+[f'{path+filename}_b{i}_{size}.tiff' for i in L8_order]+['-o', path+filename+f'_merged_{size}_true_color.tiff']
    gdal_merge.main(merge_params)


def merge_MOD09GA_refl_bands_to_tif(path, filename, size=2400, extension=".hdf"):
    """
    filename without .hdf file extension!
    """
    f= gdal.Open(path+filename+extension)
    subsets = f.GetSubDatasets()
    driver = subsets[11][0][:-3] # get common driver command # cant i just loop over drivers??
    # Reorder bands to fit the L8 order. The model is trained on this order.
    L8_order = [3, 4, 1, 2, 6, 7, 5] # [2, 3, 0, 1, 5, 6, 4] # landsat8 order of modis (modga09) bands
    # remember to +1 every band as MODIS starts its bands at index 1
    for i in L8_order:  # translate bands 1-7 to geotiff
        """ 
        Order like this:              (L8                   , MODGA09)
        fmask_config.setReflectiveBand(fmask.config.BAND_RED, 0)
        fmask_config.setReflectiveBand(fmask.config.BAND_NIR, 1)
        fmask_config.setReflectiveBand(fmask.config.BAND_BLUE, 2)
        fmask_config.setReflectiveBand(fmask.config.BAND_GREEN, 3)
        fmask_config.setReflectiveBand(fmask.config.BAND_CIRRUS, 4)
        fmask_config.setReflectiveBand(fmask.config.BAND_SWIR1, 5)
        fmask_config.setReflectiveBand(fmask.config.BAND_SWIR2, 6)
        """
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
        driveri = driver + str(i) +'_1'
        band = gdal.Open(driveri)
        gdal.Translate(destName=f'{path+filename}_b{i}_{size}.tiff', srcDS=band, format="GTiff", width=size, height=size)
    
    #gdal.Warp(destNameOrDestDS=path+filename+'_warped.tif', srcDSOrSrcDSTab=[f'{path+filename}_b{i}.tif' for i in range(1,7+1)])
    
    # merge_params = ['', '-separate']+[f'{path+filename}_b{i}.tif' for i in range(1,7+1)]+['-o', path+filename+'_merged.tif']
    merge_params = ['', '-separate']+[f'{path+filename}_b{i}_{size}.tiff' for i in L8_order]+['-o', path+filename+f'_merged_{size}.tiff']
    gdal_merge.main(merge_params)
    #!{gdal_path}gdal_merge.py -separate {modis_path+day1}_b[1-7].tif -o {modis_path+day1}_merged.tif
# SECTION: Models
