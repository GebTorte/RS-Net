import os
import time
import numpy as np
import tifffile as tiff
from PIL import Image
from srcv2_2.utils import CategoryIndexOrder, load_product, get_cls, extract_collapsed_cls, extract_cls_mask, predict_img, predict_img_v2, image_normalizer

POSITIVES = ['shadow', 'thin', 'cloud', 'snow', 'water']
NEGATIVES = ['clear']


def evaluate_test_set(model, dataset, num_gpus, params, save_output=False, write_csv=True):
    if dataset == 'Biome_gt':
        __evaluate_biome_dataset__(model, num_gpus, params, save_output=save_output, write_csv=write_csv)

    elif dataset == 'SPARCS_gt':
        __evaluate_sparcs_dataset__(model, num_gpus, params, save_output=save_output, write_csv=write_csv)


def __evaluate_sparcs_dataset__(model, num_gpus, params, save_output=False, write_csv=True):
    # Find the number of classes and bands
    if params.collapse_cls:
        n_cls = 1
    else:
        n_cls = np.size(params.cls)
    n_bands = np.size(params.bands)

    # Get the name of all the products (scenes)
    data_path = params.project_path + "data/raw/SPARCS_dataset/l8cloudmasks/sending/"
    toa_path = params.project_path + "data/processed/SPARCS_TOA/"
    products = sorted(os.listdir(data_path))
    products = [p for p in products if 'data.tif' in p]
    products = [p for p in products if 'xml' not in p]

    # If doing CV, only evaluate on test split
    if params.split_dataset:
        products = params.test_tiles[1]

    # Define thresholds and initialize evaluation metrics dict
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    if "categorical_crossentropy" in params.loss_func:
        thresholds = [params.threshold] # dummy threshold

    evaluation_metrics = {}
    evaluating_product_no = 1  # Used in print statement later

    for product in products:
        # Time the prediction
        start_time = time.time()

        # Load data
        img_all_bands = tiff.imread(data_path + product)
        img_all_bands[:, :, 0:8] = tiff.imread(toa_path + product[:-8] + 'toa.TIF')

        # Load the relevant bands and the mask
        img = np.zeros((np.shape(img_all_bands)[0], np.shape(img_all_bands)[1], np.size(params.bands)))
        for i, b in enumerate(params.bands):
            if b < 8:
                img[:, :, i] = img_all_bands[:, :, b-1]
            else:  # Band 8 is not included in the tiff file
                img[:, :, i] = img_all_bands[:, :, b-2]

        # Load true mask
        mask_true = np.array(Image.open(data_path + product[0:25] + 'mask.png'))

        # Pad the image for improved borders
        padding_size = params.overlap
        npad = ((padding_size, padding_size), (padding_size, padding_size), (0, 0))
        img_padded = np.pad(img, pad_width=npad, mode='symmetric')

        # Get the masks
        #cls = get_cls(params)
        #cls = [5]  # TODO: Currently hardcoded to look at clouds - fix it!
        cls = get_cls(params.satellite, "SPARCS_gt", params.cls)
        # mask_true_cls_corrected = np.zeros((np.shape(mask_true)[0], np.shape(mask_true)[0], len(cls)))

        # Create the binary masks
        if params.collapse_cls:
            mask_true = extract_collapsed_cls(mask_true, cls)
        else:
            for l, c in enumerate(params.cls):
                y = extract_cls_mask(mask_true, cls)

                # Save the binary masks as one hot representations
                mask_true[:, :, l] = y[:, :, 0]

        # Predict the images
        predicted_mask_padded, _ = predict_img_v2(model, params, img_padded, n_bands, n_cls, num_gpus)

        # Remove padding
        predicted_mask =  predicted_mask_padded[padding_size:-padding_size, padding_size:-padding_size, :]

        # Create a nested dict to save evaluation metrics for each product
        evaluation_metrics[product] = {}

        # Find the valid pixels and cast to uint8 to reduce processing time
        valid_pixels_mask = np.uint8(np.clip(img[:, :, 0], 0, 1))
        mask_true = np.uint8(mask_true)

        # Loop over different threshold values
        for j, threshold in enumerate(thresholds):
            predicted_binary_mask = np.uint8(predicted_mask >= threshold)

            categorical_accuracy = accuracy= omission= comission= pixel_jaccard= precision= recall= f_one_score= tp= tn = fp = fn = npix = 0
            if params.collapse_cls:
                accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix = calculate_evaluation_criteria(valid_pixels_mask, predicted_binary_mask, mask_true)
            else:
                if params.loss_func == "sparse_categorical_crossentropy":
                    categorical_accuracy, accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix = calculate_sparse_sparcs_class_evaluation_criteria(params, 
                                                                                                                                                valid_pixels_mask, predicted_mask, mask_true)
                elif params.loss_func == "categorical_crossentropy":
                    categorical_accuracy, accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix = calculate_class_evaluation_criteria(params.cls, 
                    cls, valid_pixels_mask, predicted_mask, mask_true)

            # Create an additional nesting in the dict for each threshold value
            evaluation_metrics[product]['threshold_' + str(threshold)] = {}

            # Save the values in the dict
            evaluation_metrics[product]['threshold_' + str(threshold)]['tp'] = tp
            evaluation_metrics[product]['threshold_' + str(threshold)]['fp'] = fp
            evaluation_metrics[product]['threshold_' + str(threshold)]['fn'] = fn
            evaluation_metrics[product]['threshold_' + str(threshold)]['tn'] = tn
            evaluation_metrics[product]['threshold_' + str(threshold)]['npix'] = npix
            evaluation_metrics[product]['threshold_' + str(threshold)]['categorical_accuracy'] = categorical_accuracy
            evaluation_metrics[product]['threshold_' + str(threshold)]['accuracy'] = accuracy
            evaluation_metrics[product]['threshold_' + str(threshold)]['precision'] = precision
            evaluation_metrics[product]['threshold_' + str(threshold)]['recall'] = recall
            evaluation_metrics[product]['threshold_' + str(threshold)]['f_one_score'] = f_one_score
            evaluation_metrics[product]['threshold_' + str(threshold)]['omission'] = omission
            evaluation_metrics[product]['threshold_' + str(threshold)]['comission'] = comission
            evaluation_metrics[product]['threshold_' + str(threshold)]['pixel_jaccard'] = pixel_jaccard

        print('Testing product ', evaluating_product_no, ':', product)

        exec_time = str(time.time() - start_time)
        print("Prediction finished in      : " + exec_time + "s")
        for threshold in thresholds:
            print("threshold=" + str(threshold) +
                  ": tp=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['tp']) +
                  ": fp=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['fp']) +
                  ": fn=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['fn']) +
                  ": tn=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['tn']) +
                  ": Categorical-Accuracy=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['categorical_accuracy']) +
                  ": Accuracy=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['accuracy']) +
                  ": precision=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['precision']) +
                  ": recall=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['recall']) +
                  ": omission=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['omission']) +
                  ": comission=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['comission']) +
                  ": pixel_jaccard=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['pixel_jaccard']))

        evaluating_product_no += 1

        # Save images and predictions
        data_output_path = params.project_path + "data/output/SPARCS/"
        if not os.path.isfile(data_output_path + '%s_photo.png' % product[0:24]):
            Image.open(data_path + product[0:25] + 'photo.png').save(data_output_path + '%s_photo.png' % product[0:24])
            Image.open(data_path + product[0:25] + 'mask.png').save(data_output_path + '%s_mask.png' % product[0:24])

        if save_output:
            # Save predicted mask as 16 bit png file (https://github.com/python-pillow/Pillow/issues/2970)
            arr = np.uint16(predicted_mask[:, :, 0] * 65535)
            array_buffer = arr.tobytes()
            img = Image.new("I", arr.T.shape)
            img.frombytes(array_buffer, 'raw', "I;16")
            if save_output:
                img.save(data_output_path + '%s-model%s-prediction.png' % (product[0:24], params.modelID))

            #Image.fromarray(np.uint8(predicted_mask[:, :, 0]*255)).save(data_output_path + '%s-model%s-prediction.png' % (product[0:24], params.modelID))

    exec_time = str(time.time() - start_time)
    print("Dataset evaluated in: " + exec_time + "s")
    print("Or " + str(float(exec_time)/np.size(products)) + "s per image")

    if write_csv:
        write_csv_files(evaluation_metrics, params)


def __evaluate_biome_dataset__(model, num_gpus, params, save_output=False, write_csv=True):
    """
    Evaluates all products in data/processed/visualization folder, and returns performance metrics
    """
    print('------------------------------------------')
    print("Evaluate model on visualization data set:")

    # Find the number of classes and bands
    if params.collapse_cls:
        n_cls = 1
    else:
        n_cls = np.size(params.cls)
    n_bands = np.size(params.bands)

    folders = sorted(os.listdir(params.project_path + "data/raw/Biome_dataset/"))
    folders = [f for f in folders if '.' not in f]  # Filter out .gitignore

    product_names = []

    if params.loss_func == "binary_crossentropy":
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    elif params.loss_func =="categorical_crossentropy": # for categorical argmaxing, thresholding seems to be irrelevant
        thresholds = [params.threshold]
    else:
        thresholds = [0.5]

    evaluation_metrics = {}
    evaluating_product_no = 1  # Used in print statement later

    # Used for timing tests
    load_time = []
    prediction_time = []
    threshold_loop_time = []
    save_time = []
    total_time = []

    for folder in folders:
        print('#########################')
        print('TESTING BIOME: ' + folder)
        print('#########################')
        products = sorted(os.listdir(params.project_path + "data/raw/Biome_dataset/" + folder + "/BC/"))
        
        # If doing CV, only evaluate on test split
        if params.split_dataset:
            print('NOTE: THE BIOME DATASET HAS BEEN SPLIT INTO TRAIN AND TEST')
            products = [f for f in products if f in params.test_tiles[1]]
        else:
            print('NOTE: THE ENTIRE BIOME DATASET IS CURRENTLY BEING USED FOR TEST')

        for product in products:
            print('------------------------------------------')
            print('Testing product ', evaluating_product_no, ':', product)
            data_path = params.project_path + "data/raw/Biome_dataset/" + folder + "/BC/" + product + "/"
            toa_path = params.project_path + "data/processed/Biome_TOA/" + folder + "/BC/" + product + "/"

            product_names.append(product)

            start_time = time.time()
            img, img_rgb, valid_pixels_mask = load_product(product, params, data_path, toa_path)
            load_time.append(time.time() - start_time)

            # Load the true classification mask
            mask_true = tiff.imread(data_path + product + '_fixedmask.TIF')  # The 30 m is the native resolution

            #try:
            #    if params.normalized_dataset:
            #        img = image_normalizer(img, params, 'enhance_contrast')
            #except AttributeError as e:
            #    print(e)
            #    pass

            # Get the masks
            cls = get_cls('Landsat8', 'Biome_gt', params.cls)
           
            # Create the binary masks
            if params.collapse_cls:
                print("Collapsing CLS in evaluate model.")
                mask_true = extract_collapsed_cls(mask_true, cls)
            else:
                if params.loss_func == "categorical_crossentropy":
                    # model outputs in cls order, if not otherwise (logits, labels) designed
                    pass
                    #cio = CategoryIndexOrder.CLOUD # dummy object 
                    #for l, c in enumerate(params.cls):  # depending on params.cls and cls being correctly ordered
                    #    try:
                    #        mask_true[mask_true == cls[l]] = cio.get_model_index_for_string(params.cls, c)
                    #    except IndexError as e: # index out of cls range. -> skipping this cls
                    #        print(f"IndexError on l={l}, c={c}, cls={str(cls)}", e)
                    #        continue
                elif params.loss_func == "sparse_categorical_crossentropy":
                    pass
                    
                    # mask_true = extract_collapsed_cls(mask_true, cls)
                # for l, c in enumerate(params.cls):
                #     y = extract_cls_mask(mask_true, c) # c was cls

                #     # Save the binary masks as one hot representations
                #     mask_true[:, :, l] = y[:, :, 0]

            prediction_time_start = time.time()
            predicted_mask, _ = predict_img_v2(model, params, img, n_bands, n_cls, num_gpus) # predict img_v2
            prediction_time.append(time.time() - prediction_time_start)

            # Create a nested dict to save evaluation metrics for each product
            evaluation_metrics[product] = {}

            threshold_loop_time_start = time.time()
            # mask_true = np.uint8(mask_true)
            # Loop over different threshold values
            for j, threshold in enumerate(thresholds):
                predicted_binary_mask = np.uint8(predicted_mask >= threshold)
                #predicted_mask = np.uint8(predicted_mask >= threshold) # not needed because of argmaxing i think

                categorical_cross_entropy = iou = dice_coeff=categorical_accuracy = accuracy= omission= comission= pixel_jaccard= precision= recall= f_one_score= tp= tn = fp = fn = npix = 0
                if params.collapse_cls:
                    accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix = calculate_evaluation_criteria(valid_pixels_mask.copy(), predicted_binary_mask.copy(), mask_true.copy())
                else:
                    if params.loss_func == "sparse_categorical_crossentropy" or params.loss_func == "categorical_crossentropy":
                        categorical_cross_entropy, iou, dice_coeff, categorical_accuracy, accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix = calculate_sparse_class_evaluation_criteria(params, 
                                valid_pixels_mask.copy(), predicted_mask.copy(), mask_true.copy())

                # Create an additional nesting in the dict for each threshold value
                evaluation_metrics[product]['threshold_' + str(threshold)] = {}
                # Save the values in the dict
                evaluation_metrics[product]['threshold_' + str(threshold)]['biome'] = folder  # Save biome type too
                evaluation_metrics[product]['threshold_' + str(threshold)]['tp'] = tp
                evaluation_metrics[product]['threshold_' + str(threshold)]['fp'] = fp
                evaluation_metrics[product]['threshold_' + str(threshold)]['fp'] = fp
                evaluation_metrics[product]['threshold_' + str(threshold)]['fn'] = fn
                evaluation_metrics[product]['threshold_' + str(threshold)]['tn'] = tn
                evaluation_metrics[product]['threshold_' + str(threshold)]['npix'] = npix
                evaluation_metrics[product]['threshold_' + str(threshold)]['accuracy'] = accuracy
                evaluation_metrics[product]['threshold_' + str(threshold)]['precision'] = precision
                evaluation_metrics[product]['threshold_' + str(threshold)]['recall'] = recall
                evaluation_metrics[product]['threshold_' + str(threshold)]['f_one_score'] = f_one_score
                evaluation_metrics[product]['threshold_' + str(threshold)]['omission'] = omission
                evaluation_metrics[product]['threshold_' + str(threshold)]['comission'] = comission
                evaluation_metrics[product]['threshold_' + str(threshold)]['pixel_jaccard'] = pixel_jaccard
                evaluation_metrics[product]['threshold_' + str(threshold)]['categorical_accuracy'] = categorical_accuracy
                evaluation_metrics[product]['threshold_' + str(threshold)]['categorical_cross_entropy'] = categorical_cross_entropy
                evaluation_metrics[product]['threshold_' + str(threshold)]['iou'] = iou
                evaluation_metrics[product]['threshold_' + str(threshold)]['dice_coeff'] = dice_coeff

            for threshold in thresholds:
                print("threshold=" + str(threshold) +
                        ": npix=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['npix']) +
                        ": tp=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['tp']) +
                        ": fp=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['fp']) +
                        ": fn=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['fn']) +
                        ": tn=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['tn']) +
                        ": Categorical-accuracy=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['categorical_accuracy']) +
                        ": IoU=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['iou']) +
                        ": Dice Coefficient=" +   str(evaluation_metrics[product]['threshold_' + str(threshold)]['dice_coeff']) +
                        ": Categorical-cross-entropy=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['categorical_cross_entropy']) +
                        ": Accuracy=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['accuracy']) +
                        ": precision=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['precision'])+
                        ": recall=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['recall']) +
                        ": omission=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['omission']) +
                        ": comission=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['comission'])+
                        ": pixel_jaccard=" + str(evaluation_metrics[product]['threshold_' + str(threshold)]['pixel_jaccard']))

            threshold_loop_time.append(time.time() - threshold_loop_time_start)

            evaluating_product_no += 1
            
            save_time_start = time.time()

            data_output_path = params.project_path + "data/output/Biome/"
            if not os.path.isfile(data_output_path + '%s-photo.png' % product):
                img_enhanced_contrast = image_normalizer(img_rgb, params, type='enhance_contrast')
                Image.fromarray(np.uint8(img_enhanced_contrast * 255)).save(data_output_path + '%s-photo.png' % product)
                Image.open(data_path + product + '_fixedmask.TIF').save(data_output_path + '%s-mask.png' % product)

            if save_output:
                # Save images and predictions
                os.makedirs(data_output_path + params.modelID, exist_ok=True) # make folder for model

                # Save predicted mask as 16 bit png file (https://github.com/python-pillow/Pillow/issues/2970)
                #arr = np.uint16(predicted_mask[:, :, 0] * (2**16-1))
                

                if params.loss_func == "sparse_categorical_crossentropy":
                    argmaxed_pred = np.argmax(predicted_mask, axis=-1)
                    #predicted_mask_copy = predicted_mask.copy() # this would be the multi-layered output. Too big as tiff though.
                    # get_cls(params.satellite, params.train_dataset, params.cls)
                    for i, c in enumerate(get_cls(params.satellite, params.test_dataset, params.cls)): # cls have to be converted by get_cls beforehand
                        argmaxed_pred[argmaxed_pred == i] = min(c, 2**8 - 1) # as c is uint8
                        #predicted_mask_copy[:,:,i][argmaxed_pred == i] = c

                    img = Image.fromarray(np.uint8(argmaxed_pred))
                    #arr2_buffer = np.uint16(argmaxed_pred).tobytes()
                    #img2 = Image.new("RGB", argmaxed_pred.T.shape)
                    #img2.frombytes(arr2_buffer, 'raw', "I;16")
                    img.save(data_output_path + params.modelID + f'/{product}-nb_prediction.png')
                    
                    # predicted_mask_copy_buffer = np.uint8(predicted_mask_copy).tobytes()
                    #img3 = Image.fromarray(predicted_mask_copy, mode='RGBA')
                    #img3.save(data_output_path + params.modelID + f'/{product}-layered_nb_prediction.png')

                    #deactivate for now as tiffs use too much space
                    #tiff.imwrite(data_output_path + params.modelID + f'/{product}-layered_nb_prediction.tiff', data=np.uint8(predicted_mask_copy))
                    #tiff.imwrite(data_output_path + params.modelID + f'/{product}-nb_prediction.tiff', data=arr2_buffer)
                try:
                    cloud_arr = np.uint8(predicted_mask[:, :, params.cls.index('cloud')] * (2**8-1)) # indexed layer of cloud, assuming cloud is in predicted classes
                    #array_buffer = cloud_arr.tobytes()

                    img = Image.fromarray(cloud_arr)
                    #img = Image.new("L", cloud_arr.T.shape)
                    #img.frombytes(array_buffer, 'raw', "L")
                    img.save(data_output_path + params.modelID + f'/{product}-cloud_prediction.png')
                except:
                    pass
            save_time.append(time.time() - save_time_start)

            #Image.fromarray(np.uint16(predicted_mask[:, :, 0] * 65535)).\
            #    save(data_output_path + '%s-model%s-prediction.png' % (product, params.modelID))

            total_time.append(time.time() - start_time)
            print("Data loaded in                       : " + str(load_time[-1]) + "s")
            print("Prediction finished in               : " + str(prediction_time[-1]) + "s")
            print("Threshold loop finished in           : " + str(threshold_loop_time[-1]) + "s")
            print("Results saved in                     : " + str(save_time[-1]) + "s")
            print("Total time for product finished in   : " + str(total_time[-1]) + "s")

    # Print timing results
    print("Timing analysis for Biome dataset:")
    print("Load time: mean val.=" + str(np.mean(load_time)) + ", with std.=" + str(np.std(load_time)))
    print("Prediction time: mean val.=" + str(np.mean(prediction_time)) + ", with std.=" + str(np.std(prediction_time)))
    print("Threshold loop time: mean val.=" + str(np.mean(threshold_loop_time)) + ", with std.=" + str(np.std(threshold_loop_time)))
    print("Save time: mean val.=" + str(np.mean(save_time)) + ", with std.=" + str(np.std(save_time)))
    print("Total time: mean val.=" + str(np.mean(total_time)) + ", with std.=" + str(np.std(total_time)))

    # The mean jaccard index is not a weighted average of the number of pixels, because the number of pixels in the
    # product is dependent on the angle of the product. I.e., if the visible pixels are tilted 45 degrees, there will
    # be a lot of NaN pixels. Hence, the number of visible pixels is constant for all products.
    # for i, threshold in enumerate(thresholds):
    #     params.threshold = threshold  # Used when writing the csv files
    #     write_csv_files(np.mean(pixel_jaccard[i, :]), pixel_jaccard[i, :], product_names, params)
    if write_csv:
        write_csv_files(evaluation_metrics, params)

def calculate_sparse_sparcs_class_evaluation_criteria(params, valid_pixels_mask, predicted_mask, true_mask):
    """
    deprecated
    ATTENTION: this is only for non-collapse-cls
    """
    print("Sparse Metrics")

    cls = get_cls(params.satellite, params.train_dataset, params.cls)

    true_mask_cls_corrected = true_mask.copy()
    for i, c in enumerate(cls):
        true_mask_cls_corrected[true_mask_cls_corrected == i] = c
    #i = -1
    #for l, c in enumerate(cls):
    #    i += 1
    #    if c == 1 or c == 6: # skipping combined classes
    #        i-=1
    #        continue#
#
#                y = extract_cls_mask(mask_true, c)

        # For Sparcs gt combine classes (0,1) (2, 6)
        """
        if c == 'shadow':
            cls_int.append(0)
            cls_int.append(1)
        elif c == 'water':
            cls_int.append(2)
            cls_int.append(6)
        elif c == 'snow':
            cls_int.append(3)
        elif c == 'cloud':
            cls_int.append(5)
        elif c == 'clear':
            cls_int.append(4)
        """

        # ATTENTION: depending on int_cls input order in ImageSequence, model output order may vary...  
        #if c == 0:
        #    y |= extract_cls_mask(mask_true, 1)
        #elif c == 2:
        #    y |= extract_cls_mask(mask_true, 6)

        # Save the binary masks as one hot representations
        #mask_true_cls_corrected[:,:,i] = y

    # Count number of actual pixels
    npix = valid_pixels_mask.sum()
    valid_pixels_mask = np.asarray(valid_pixels_mask, dtype=bool)
 
    # TODO: combine classes (0,1) and (2,6) in predicted mask and lAND it with true_mask (cls-corrected one)
            # Create the binary masks
    cls = get_cls(params.satellite, params.train_dataset, params.cls)
    predicted_mask_cls_corrected = predicted_mask.copy()

    predicted_mask_cls_corrected = np.argmax(predicted_mask_cls_corrected, axis=-1)

    for i, c in enumerate(cls):
        predicted_mask_cls_corrected[predicted_mask_cls_corrected == i] = c # ...

    #i = -1
    #for l, c in enumerate(cls):
    #    i+=1
    #    if c == 1 or c == 6: # skipping combined classes
    #        i-=1
    #        continue#
#
#        y = extract_cls_mask(predicted_mask_cls_corrected, c)#

        # For Sparcs gt combine classes (0,1) (2, 6)

        # ATTENTION: depending on int_cls input order in ImageSequence, model output order may vary...  
#        if c == 0:
#            y |= extract_cls_mask(predicted_mask_cls_corrected, 1)
#        elif c == 2:
#            y |= extract_cls_mask(predicted_mask_cls_corrected, 6)

        # Save the binary masks as one hot representations
#        predicted_mask_cls_corrected[:, :, i] = y
    true_mask_cls_corrected = np.uint8(true_mask_cls_corrected)
    categorical_accuracy = (predicted_mask_cls_corrected & true_mask_cls_corrected) / npix  # (true mask cls corrected has size(dim3) = 7, predicted mask not )
    
    positives_mask = get_cls(params.satellite, params.train_dataset, cls_string=POSITIVES)

    #non-cloudy types
    # [x for x in ['fill', 'clear'] if x in enumeration_cls]
    negatives_mask = get_cls(params.satellite, params.train_dataset, cls_string=NEGATIVES)

    # perhaps index-correct the masks before
    pred_positives = np.isin(predicted_mask, positives_mask)
    pred_negatives = np.isin(predicted_mask, negatives_mask)
    true_positives = np.isin(true_mask, positives_mask)
    true_negatives =  np.isin(true_mask, negatives_mask)

    # npix = npix*n_cls ???

    tp = ((pred_positives & true_positives) & valid_pixels_mask).sum()
    fp = (true_negatives & pred_positives & valid_pixels_mask).sum()
    fn = ((pred_negatives & true_positives) & valid_pixels_mask).sum()
    tn = npix - tp - fp - fn

    # Calculate metrics
    accuracy = (tp + tn) / npix
    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_one_score = 2 * (precision * recall) / (precision + recall)
        # See https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
        pixel_jaccard = tp / (tp + fp + fn)
    else:
        precision = recall = f_one_score = pixel_jaccard = 0

    # Metrics from Foga 2017 paper
    # if fp!=0: # accunting for runtime division by 0 (tn was 0)
    if fp != 0 and tn != 0:
        omission = fp / (tp + fp)
        comission = fp / (tn + fn)

    else:
        omission = comission = 0

    return categorical_accuracy, accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix

def calculate_sparse_class_evaluation_criteria_v2(params, valid_pixels_mask, predicted_mask, mask_true):
    """
    deprecated
    """
    print("Sparse Metrics")
    # Count number of actual pixels
    npix = valid_pixels_mask.sum()
    valid_pixels_mask = np.asarray(valid_pixels_mask, dtype=bool)
    invalid_pixels_mask = ~valid_pixels_mask
    #n_invalid_pix = invalid_pixels_mask.sum()
    # converted mask_true to predicted values as input
    #mask_true_cls_corrected = true_mask.copy()

    argmaxed_pred_mask = np.argmax(predicted_mask, axis=-1)

    cls = get_cls(params.satellite, params.test_dataset, params.cls)
    
    #for i, c in enumerate(get_cls(params.satellite, params.test_dataset, params.cls)): # cls have to be converted by get_cls beforehand# has to be correct order!
    #    argmaxed_pred_mask[argmaxed_pred_mask == i] = c  # convert indices of model output to cls

    # all layers, since true mask contains 
    true_mask_cls = get_cls(params.satellite, params.test_dataset, POSITIVES + NEGATIVES + ['fill']) 

    drop_indices = []
    valid_indices = []
    mask_true_index_corrected = mask_true # & valid_pixels_mask
    for i, c in enumerate(cls): # cls have to be converted by get_cls beforehand# has to be correct order!
        drop_indices.append(true_mask_cls.index(c))
        valid_indices.append(cls.index())
        mask_true_index_corrected[mask_true_index_corrected == c] = i 

    drop_indices = set(drop_indices + valid_indices)

    mask_true_index_corrected = np.delete(mask_true_index_corrected, ) # delete invalid axes from mask true (before scale up to 3d)
    # valid_argmaxed_pred_mask = np.uint8(argmaxed_pred_mask) & valid_pixels_mask
    #valid_true_mask = np.uint8(mask_true) & valid_pixels_mask

    binary_accuracy_mask = argmaxed_pred_mask == mask_true_index_corrected # &?
    binary_accuracy_mask &= valid_pixels_mask # remote invalid pixel
    equal_count=binary_accuracy_mask.sum()

    #categorical_accuracy = # of correctly predicted records / total number of records
    # if fill is in predicted classes, this value will automatically (falsely) be higher
    categorical_accuracy = equal_count / npix

    # perhaps implement acc,pred,... for every cls type
    #cloudy types / classy types
    positives = get_cls(params.satellite, params.train_dataset, cls_string=POSITIVES)

    #non-cloudy types
    negatives = get_cls(params.satellite, params.train_dataset, cls_string=NEGATIVES) # 'fill', removed fill as it should not count into metrics

    pred_positives = np.isin(argmaxed_pred_mask, positives)
    pred_negatives = np.isin(argmaxed_pred_mask, negatives)
    true_positives = np.isin(mask_true, positives)
    true_negatives =  np.isin(mask_true, negatives)

    #pred_positives_sum = pred_positives.sum()
    #pred_negatives_sum = pred_negatives.sum()
    #true_positives_sum = true_positives.sum()
    #true_negatives_sum = true_negatives.sum()

    tp = ((pred_positives & true_positives) & valid_pixels_mask).sum() 
    fp = (true_negatives & pred_positives & valid_pixels_mask).sum() 
    fn = ((pred_negatives & true_positives) & valid_pixels_mask).sum() 
    tn = npix - tp - fp - fn

    # Calculate metrics
    accuracy = (tp + tn) / npix
    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_one_score = 2 * (precision * recall) / (precision + recall)
        # See https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
        pixel_jaccard = tp / (tp + fp + fn)
    else:
        precision = recall = f_one_score = pixel_jaccard = 0

    # Metrics from Foga 2017 paper
    # if fp!=0: # accunting for runtime division by 0 (tn was 0)
    if fp != 0 and tn != 0:
        omission = fp / (tp + fp)
        comission = fp / (tn + fn)

    else:
        omission = comission = 0

    return categorical_accuracy, accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix

def calculate_categorical_accuracy(y_true, y_pred, valid_pixel_mask, cls, npix):
    """
    y_true and y_pred contain classes from cls
    """
    cat_accs = []
    for c in cls:
        if c == 0: # pass on fill pixel
            continue
        true_cls = (y_true == c) & valid_pixel_mask
        pred_cls = (y_pred == c) & valid_pixel_mask
        cat_accs.append(np.sum(true_cls==pred_cls))
    return np.mean(cat_accs) / (npix * len(cat_accs))

def calculate_dice_coefficient(y_true, y_pred, valid_pixel_mask, cls):
    dice_scores = []
    for c in cls:
        if c == 0:
            continue
        true_cls = (y_true == c) & valid_pixel_mask
        pred_cls = (y_pred == c) & valid_pixel_mask
        intersection = np.sum(true_cls * pred_cls)
        union = np.sum(true_cls | pred_cls) #  np.sum(true_cls) + np.sum(pred_cls)
        dice_coefficient = (2. * intersection) / union
        dice_scores.append(dice_coefficient)
    return np.mean(dice_scores)

def calculate_iou(y_true, y_pred, valid_pixel_mask, cls):
    iou_scores = []
    for c in cls:
        if c == 0:
            continue
        true_cls = (y_true == c)& valid_pixel_mask
        pred_cls = (y_pred == c)& valid_pixel_mask
        intersection = true_cls & pred_cls
        union = true_cls | pred_cls
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)
    return np.mean(iou_scores)

def calculate_categorical_cross_entropy(y_true, y_pred, cls):
    """
    y_true of cls
    y_pred of probabilities
    """
    # one hot encode y_true
    shp = np.shape(y_true)
    one_hot = np.zeros(shape=(shp[0], shp[1], len(cls)))

    for i, c in enumerate(cls):
        if c == 0:
            continue
        one_hot[:,:,i] = (y_true == c).astype(np.uint8)

    # ensure probabilities per pixel sum up to 1
    y_pred /= np.sum(y_pred, axis=-1, keepdims=True)

    epsilon = 1e-15  # Small constant to avoid division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    return -np.mean(np.sum(one_hot * np.log(y_pred), axis=-1))

def calculate_sparse_class_evaluation_criteria(params, valid_pixels_mask, predicted_mask, mask_true):
    """
    Here be bugs
    """
    print("Sparse Metrics")
    # Count number of actual pixels # should not be needed for only bands 1-7
    valid_pixels_mask = np.asarray(valid_pixels_mask, dtype=bool)
    npix = valid_pixels_mask.sum()
    #invalid_pixels_mask = ~valid_pixels_mask


    enumeration_cls = get_cls(params.satellite, params.train_dataset, params.cls)
    
    #fill_pixel_mask = mask_true == get_cls(params.satellite, params.test_dataset, ['fill'])[0] 
    #fill_and_valid_pixels_mask = fill_pixel_mask & valid_pixels_mask

    # npix = valid_pixels_mask.sum() - fill_and_valid_pixels_mask.sum()
    #n_invalid_pix = invalid_pixels_mask.sum()
    # converted mask_true to predicted values as input
    #mask_true_cls_corrected = true_mask.copy()
    argmaxed_pred_mask =  np.argmax(predicted_mask, axis=-1)
    argmaxed_cls_pred_mask = argmaxed_pred_mask.copy()
    for i, c in enumerate(enumeration_cls): # cls have to be converted by get_cls beforehand# has to be correct order!
        argmaxed_cls_pred_mask[argmaxed_cls_pred_mask == i] = c  # convert indices of model output to cls

    binary_accuracy_mask = argmaxed_cls_pred_mask == mask_true
    binary_accuracy_mask &= valid_pixels_mask
    # binary_accuracy_mask &= ~fill_pixel_mask # remove fill pixel
    equal_count=binary_accuracy_mask.sum()

    # categorical_accuracy = # of correctly predicted records / total number of records
    categorical_accuracy = equal_count / npix # (npix - fill_and_valid_pixels_mask.sum()) #, valid_pixels_mask.sum()) # - fill_and_valid_pixels_mask.sum(
    print("old categorical_accuracy (with fill pxl?!): ", categorical_accuracy)
    
    # perhaps implement acc,pred,... for every cls type
    #cloudy types / classy types
    positives = get_cls(params.satellite, params.train_dataset, cls_string=POSITIVES)

    #non-cloudy types
    negatives = get_cls(params.satellite, params.train_dataset, cls_string=NEGATIVES) # 'fill', removed fill as it should not count into metrics

    # The following are disregarding fill by default. As it is not in negatives nor positives
    pred_positives = np.isin(argmaxed_cls_pred_mask, positives)
    pred_negatives = np.isin(argmaxed_cls_pred_mask, negatives)
    true_positives =  np.isin(mask_true, positives)
    true_negatives =  np.isin(mask_true, negatives)

    tp = ((pred_positives & true_positives) & valid_pixels_mask).sum() 
    fp = (true_negatives & pred_positives & valid_pixels_mask).sum() 
    fn = ((pred_negatives & true_positives) & valid_pixels_mask).sum() 
    tn = (pred_negatives & true_negatives & valid_pixels_mask).sum()
    #tn = npix - tp - fp - fn

    # Calculate metrics
    accuracy = (tp + tn) / npix
    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_one_score = 2 * (precision * recall) / (precision + recall)
        # See https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
        pixel_jaccard = tp / (tp + fp + fn)
    else:
        precision = recall = f_one_score = pixel_jaccard = 0

    # Metrics from Foga 2017 paper
    # if fp!=0: # accunting for runtime division by 0 (tn was 0)
    if fp != 0 and tn != 0:
        omission = fp / (tp + fp)
        comission = fp / (tn + fn)

    else:
        omission = comission = 0

    iou = calculate_iou(mask_true.copy(), argmaxed_cls_pred_mask.copy(), valid_pixels_mask, enumeration_cls)
    dice_coeff = calculate_dice_coefficient(mask_true.copy(), argmaxed_cls_pred_mask.copy(), valid_pixels_mask,enumeration_cls)
    categorical_accuracy = calculate_categorical_accuracy(mask_true.copy(), argmaxed_cls_pred_mask.copy(), valid_pixels_mask, enumeration_cls, npix)
    categorical_cross_entropy = calculate_categorical_cross_entropy(mask_true.copy(), predicted_mask.copy(), enumeration_cls)

    return categorical_cross_entropy, iou, dice_coeff, categorical_accuracy, accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix


def calculate_class_evaluation_criteria(param_cls, cls, valid_pixels_mask, predicted_mask, true_mask):
    """
    deprecated
    """
    # Count number of actual pixels
    npix = valid_pixels_mask.sum()

    # converted mask_true to predicted values as input
    mask_true_cls_corrected = true_mask.copy()
                
    # argmax over predicted masks
    # convert index to type
    # see if index-type corresponds to true_mask category type -> if yes, considered accurate
    argmaxed_pred_mask = np.argmax(predicted_mask, axis=-1)

    binary_accuracy_mask = argmaxed_pred_mask == mask_true_cls_corrected
    binary_accuracy_mask &= np.asarray(valid_pixels_mask, dtype=bool) # remote invalid pixel
    equal_count=np.sum(binary_accuracy_mask)

    #categorical_accuracy = # of correctly predicted records / total number of records
    categorical_accuracy = equal_count / npix

    #cloudy types
    positives_mask = [CategoryIndexOrder.THIN, CategoryIndexOrder.CLOUD, CategoryIndexOrder.SHADOW]
    positives_mask = [c.get_model_index_for_type(param_cls, c) for c in positives_mask]
    positives_mask = [x for x in positives_mask if x is not None]

    #non-cloudy types
    negatives_mask = [CategoryIndexOrder.CLEAR, CategoryIndexOrder.SNOW, CategoryIndexOrder.WATER, CategoryIndexOrder.FILL]
    negatives_mask = [c.get_model_index_for_type(param_cls, c) for c in negatives_mask]
    negatives_mask = [x for x in negatives_mask if x is not None]

    # this might not run correctly if positives and negatives indices are off!
    # perhaps index-correct the masks before
    pred_positives = np.isin(argmaxed_pred_mask, positives_mask)
    pred_negatives = np.isin(argmaxed_pred_mask, negatives_mask)
    true_positives = np.isin(mask_true_cls_corrected, positives_mask)

    tp = ((pred_positives & true_positives) & valid_pixels_mask).sum()
    fp = (pred_positives & np.isin(mask_true_cls_corrected, negatives_mask) & valid_pixels_mask).sum()
    fn = ((pred_negatives & true_positives) & valid_pixels_mask).sum()
    tn = npix - tp - fp - fn

    # Calculate metrics
    accuracy = (tp + tn) / npix
    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_one_score = 2 * (precision * recall) / (precision + recall)
        # See https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
        pixel_jaccard = tp / (tp + fp + fn)
    else:
        precision = recall = f_one_score = pixel_jaccard = 0

    # Metrics from Foga 2017 paper
    # if fp!=0: # accunting for runtime division by 0 (tn was 0)
    if fp != 0 and tn != 0:
        omission = fp / (tp + fp)
        comission = fp / (tn + fn)

    else:
        omission = comission = 0

    return categorical_accuracy, accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix


def calculate_evaluation_criteria(valid_pixels_mask, predicted_binary_mask, true_binary_mask):
    # From https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly
    # with correction for toggling true/false from
    # https://stackoverflow.com/questions/39164786/invert-0-and-1-in-a-binary-array
    # Need to AND with the a mask showing where there are pixels to avoid including pixels with value=0

    # Count number of actual pixels
    npix = valid_pixels_mask.sum()

    if np.ndim(predicted_binary_mask) == 3:
        tp = ((predicted_binary_mask[:, :, 0] & true_binary_mask) & valid_pixels_mask).sum()
        fp = ((predicted_binary_mask[:, :, 0] & (1 - true_binary_mask)) & valid_pixels_mask).sum()
        fn = (((1 - predicted_binary_mask)[:, :, 0] & true_binary_mask) & valid_pixels_mask).sum()
        #tn = (((1 - predicted_binary_mask)[:, :, 0] & (1 - true_binary_mask)) & actual_pixels_mask).sum()
        tn = npix - tp - fp - fn
        
    else:
        tp = ((predicted_binary_mask & true_binary_mask) & valid_pixels_mask).sum()
        fp = ((predicted_binary_mask & (1 - true_binary_mask)) & valid_pixels_mask).sum()
        fn = (((1 - predicted_binary_mask) & true_binary_mask) & valid_pixels_mask).sum()
        #tn = (((1 - predicted_binary_mask) & (1 - true_binary_mask)) & actual_pixels_mask).sum()
        tn = npix - tp - fp - fn

    # Calculate metrics
    accuracy = (tp + tn) / npix
    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_one_score = 2 * (precision * recall) / (precision + recall)
        # See https://en.wikipedia.org/wiki/Jaccard_index#Similarity_of_asymmetric_binary_attributes
        pixel_jaccard = tp / (tp + fp + fn)
    else:
        precision = recall = f_one_score = pixel_jaccard = 0

    # Metrics from Foga 2017 paper
    # if fp!=0: # accunting for runtime division by 0 (tn was 0)
    if fp != 0 and tn != 0:
        omission = fp / (tp + fp)
        comission = fp / (tn + fn)

    else:
        omission = comission = 0

    return accuracy, omission, comission, pixel_jaccard, precision, recall, f_one_score, tp, tn, fp, fn, npix


def __evaluate_sentinel2_dataset__(model, num_gpus, params):
    test_data_path = params.project_path + 'data/processed/visualization/'
    files = sorted(os.listdir(test_data_path))  # os.listdir loads in arbitrary order, hence use sorted()
    files = [f for f in files if 'B02_10m.tiff' in f]  # Filter out one ID for each tile

    print('-----------------------------------------------------------------------------------------------------')
    for i, file in enumerate(files):
        print('Evaluating tile (', i + 1, 'of', np.size(files), ') :', file[0:26])
        file = file[0:26]
        img, _ = load_product(file, params, test_data_path)
        # NOTE: Needs the masks
        print('---')


def write_csv_files(evaluation_metrics, params):
    file_name = f"param_optimization_{params.train_dataset}_Train_{params.test_dataset}_Eval.csv"
    # if 'Biome' in params.train_dataset and 'Biome' in params.test_dataset:
    #     file_name = 'param_optimization_BiomeTrain_BiomeEval.csv'
    # elif 'SPARCS' in params.train_dataset and 'SPARCS' in params.test_dataset:
    #     file_name = 'param_optimization_SPARCSTrain_SPARCSEval.csv'
    # elif 'Biome' in params.train_dataset and 'SPARCS' in params.test_dataset:
    #     file_name = 'param_optimization_BiomeTrain_SPARCSEval.csv'
    # elif 'SPARCS' in params.train_dataset and 'Biome' in params.test_dataset:
    #     file_name = 'param_optimization_SPARCSTrain_BiomeEval.csv'

    if 'fmask' in params.train_dataset:
        file_name = file_name[:-4] + '_fmask.csv'

    # Create csv file
    if not os.path.isfile(params.project_path + 'reports/Unet/' + file_name):
        f = open(params.project_path + 'reports/Unet/' + file_name, 'a')

        # Create headers for parameters
        string = 'modelID,'
        for key in params.keys():
            if key == 'modelID':
                pass
            elif key == 'test_tiles':
                pass
            else:
                string += key + ','

        # Create headers for evaluation metrics
        for i, product in enumerate(list(evaluation_metrics)):  # Lists all product names
            # Add product name to string
            string += 'tile_' + str(i) + ','

            # Only need examples from one threshold key (each threshold is a new line in the final csv file)
            threshold_example_key = list(evaluation_metrics[product])[0]
            for key in list(evaluation_metrics[product][threshold_example_key]):
                string += key + '_' + str(i) + ','

        # Create headers for averaged metrics
        f.write(string + 'mean_accuracy,mean_precision,mean_recall,mean_f_one_score,mean_omission,mean_comission,mean_pixel_jaccard,mean_categorical_accuracy,mean_iou,mean_dice_coefficient,mean_categorical_cross_entropy\n')
        f.close()

    # Write a new line for each threshold value
    for threshold in list(evaluation_metrics[list(evaluation_metrics)[0]]):  # Use first product to list thresholds
        # Update the params threshold value before writing
        # params.threshold = str(threshold[-3:])
        f = open(params.project_path + 'reports/Unet/' + file_name, 'a')

        # Write params values
        string = str(params.modelID) + ','
        for key in params.values().keys():
            if key == 'modelID':
                continue
            elif key == 'test_tiles':
                continue
            elif key == 'cls' or key=='int_cls' or key=='str_cls':
                addition = ("".join(str(c) for c in params.values()[key])) 
            elif key == 'bands':
                addition = ("".join(str(b) for b in params.values()[key])) 
            else:
                addition = str(params.values()[key])

            string += str(addition).replace(",", "|")  + ',' # cant have extra commata in csv

        # Initialize variables for calculating mean visualization set values
        cat_cross_entropy_sum = dice_coeff_sum = iou_sum = categorical_accuracy_sum = accuracy_sum = precision_sum = recall_sum = f_one_score_sum = omission_sum = comission_sum = pixel_jaccard_sum=0.0

        # Write visualization set values
        for product in list(evaluation_metrics):
            # Add product name to string
            string += product + ','
            for key in (evaluation_metrics[product][threshold]):
                # Add values to string
                string += str(evaluation_metrics[product][threshold][key]) + ','

                # Extract values for calculating mean values of entire visualization set
                if 'categorical_accuracy' in key: # this has to be run before accuracy summation, as the accuracy if statement holds for categorical_accuracy aswell...
                    categorical_accuracy_sum = categorical_accuracy_sum + evaluation_metrics[product][threshold][key]    
                elif 'accuracy' in key:
                    accuracy_sum += evaluation_metrics[product][threshold][key]
                elif 'precision' in key:
                    precision_sum += evaluation_metrics[product][threshold][key]
                elif 'recall' in key:
                    recall_sum += evaluation_metrics[product][threshold][key]
                elif 'f_one_score' in key:
                    f_one_score_sum += evaluation_metrics[product][threshold][key]
                elif 'omission' in key:
                    omission_sum += evaluation_metrics[product][threshold][key]
                elif 'comission' in key:
                    comission_sum += evaluation_metrics[product][threshold][key]
                elif 'pixel_jaccard' in key:
                    pixel_jaccard_sum += evaluation_metrics[product][threshold][key]
                elif 'dice_coeff' in key:
                    dice_coeff_sum += evaluation_metrics[product][threshold][key]
                elif key == 'iou':
                    iou_sum += evaluation_metrics[product][threshold][key]
                elif key == "categorical_cross_entropy":
                    cat_cross_entropy_sum += evaluation_metrics[product][threshold][key]

        # Add mean values to string
        n_products = np.size(list(evaluation_metrics))
        string += str(accuracy_sum / n_products) + ',' + str(precision_sum / n_products) + ',' + \
                  str(recall_sum / n_products) + ',' + str(f_one_score_sum / n_products) + ',' + \
                  str(omission_sum / n_products) + ',' + str(comission_sum / n_products) + ',' + \
                  str(pixel_jaccard_sum / n_products)+ ',' + str(categorical_accuracy_sum / n_products) + ',' + \
                  str(iou_sum / n_products) + ','+ str(dice_coeff_sum / n_products) + ',' + \
                  str(cat_cross_entropy_sum/n_products)

        
        # Write string and close csv file
        f.write(string + '\n')
        f.close()