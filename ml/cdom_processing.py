__author__ = 'Gonzalo Mateo-García, Ana Ruescas'
import numpy as np
import os
import spectral.io.envi as envi
import xml.etree.ElementTree as emt

RELATION_BAND_NAMES_C2X = {'400': '01',
 '412.5': '02',
 '442.5': '03',
 '490': '04',
 '510': '05',
 '560': '06',
 '620': '07',
 '665': '08',
 '673.75': '09',
 '681.25': '10',
 '708.75': '11',
 '753.75': '12',
 '778.75': '16',
 '865': '17',
 '885': '18',
 'S3ratio1': 'S3ratio1',
 'S3ratio2': 'S3ratio2'}

RELATION_BAND_NAMES_C2RCC = {'400': '1',
 '412.5': '2',
 '442.5': '3',
 '490': '4',
 '510': '5',
 '560': '6',
 '620': '7',
 '665': '8',
 '673.75': '9',
 '681.25': '10',
 '708.75': '11',
 '753.75': '12',
 '778.75': '16',
 '865': '17',
 '885': '18',
 '1020': '21',
 'S3ratio1': 'S3ratio1',
 'S3ratio2': 'S3ratio2'}

RELATION_BAND_NAMES_SYKE = {'S3400': '01',
 'S3412.5': '02',
 'S3442.5': '03',
 'S3490': '04',
 'S3510': '05',
 'S3560': '06',
 'S3620': '07',
 'S3665': '08',
 'S3673.75': '09',
 'S3681.25': '10',
 'S3708.75': '11',
 'S3753.75': '12',
 'S3778.75': '16',
 'S3865': '17',
 'S3885': '18',
 'S3ratio1': 'S3ratio1',
 'S3ratio2': 'S3ratio2'}


BAND_NAMES_S3 = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '16', '17', '18', '21']
BAND_NAMES_S3_C2RCC = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '16', '17','18', '21']
BAND_NAMES_S3_RATIOS = BAND_NAMES_S3 + ['S3ratio1','S3ratio2']
BAND_NAMES_S3_RATIOS_C2RCC = BAND_NAMES_S3_C2RCC + ['S3ratio1','S3ratio2']

def image_to_predict_S3(img,bands_dataset,relation=RELATION_BAND_NAMES_C2X):
    """
    img: 3D np.array
    band_names:  list with the names of the bands of img
    bands_name: "S3bands","S3bands&ratios", "ratios_S3", "S3ratio1"...
    """
    bands_predict = [relation[b] for b in bands_dataset]
    index_bands_predict = [BAND_NAMES_S3_RATIOS.index(b) for b in bands_predict]
    return img[:,:,index_bands_predict]

def image_to_predict_S3_C2RCC(img,bands_dataset,relation=RELATION_BAND_NAMES_C2RCC):
    """
    img: 3D np.array
    band_names:  list with the names of the bands of img
    bands_name: "S3bands","S3bands&ratios", "ratios_S3", "S3ratio1"...
    """
    bands_predict = [relation[b] for b in bands_dataset]
    index_bands_predict = [BAND_NAMES_S3_RATIOS_C2RCC.index(b) for b in bands_predict]
    return img[:,:,index_bands_predict]


def load_S3_image(image_path):
    """
    load S3 image downloaded with SNAP, applies conversion to TOA and compute ratios

    :param image_path: path to .data/ dir
    :return: 3D np.array with the image bands sorted in BAND_NAMES_S3_RATIOS order
    """
    inputs = [os.path.join(image_path,"Oa%s_reflectance.hdr"%b) for b in BAND_NAMES_S3]


    images = []
    for ip in inputs:
        metadata = envi.read_envi_header(ip)
        gain = np.array([float(m) for m in metadata['data gain values']])
        offset = np.array([float(m) for m in metadata['data offset values']])
        img = np.float64(envi.open(ip)[:,:,:])
        img *= gain
        img += offset
        img/= np.pi
        images.append(img)

    S3ratio1 = (images[7]) / (images[3])
    S3ratio2 = (images[10]) / (images[3])
    images.append(S3ratio1)
    images.append(S3ratio2)
    images = np.concatenate(images,axis=2)

    return images

def load_S3_image_C2RCC(image_path):
    """
    load S3 image downloaded with SNAP, applies conversion to TOA and compute ratios
    C2RCC does not need conversion to Rrs

    :param image_path: path to .data/ dir
    :return: 3D np.array with the image bands sorted in BAND_NAMES_S3_RATIOS order
    """

    inputs = [os.path.join(image_path, "rrs_%s.hdr" % b) for b in BAND_NAMES_S3_C2RCC]
    # print(inputs)

    images = []
    for ip in inputs:
        metadata = envi.read_envi_header(ip)
        gain = np.array([float(m) for m in metadata['data gain values']])
        offset = np.array([float(m) for m in metadata['data offset values']])
        img = np.float64(envi.open(ip)[:,:,:])
        img *= gain
        img += offset
        images.append(img)


    S3ratio1 = (images[7]) / (images[3]+1e-5)
    S3ratio2 = (images[10]) / (images[3]+1e-5)
    images.append(S3ratio1)
    images.append(S3ratio2)
    images = np.concatenate(images,axis=2)

    return images

def load_mask(image_path):
    """
    laod the water mask from S3 image

    :param image_path: path to .data/ dir
    :return:
    """
    mask_file = os.path.join(image_path,"WQSF_lsb.hdr")
    mask = envi.open(mask_file)[:,:,0]
    mask = mask[:,:,0]
    dim_file,_ = os.path.splitext(image_path)
    dim_file+=".dim"

    e = emt.parse(dim_file).getroot()

    gp = e.find("./Flag_Coding[@name='WQSF_lsb']")
    flag_to_index = dict([(f.find("Flag_Name").text,int(f.find("Flag_Index").text)) for f in gp.getchildren()])
    mascara = np.zeros(mask.shape,dtype=mask.dtype)

    masks_to_use = ["CLOUD","CLOUD_AMBIGUOUS","CLOUD_MARGIN","SNOW_ICE","HIGHGLINT"]
    for m in masks_to_use:
        mascara |= ((mask & flag_to_index[m])!=0)

    mascara = (mascara==1)

    masks_to_use = ["WATER","INLAND_WATER"]
    mascara_water = np.zeros(mask.shape,dtype=mask.dtype)
    for m in masks_to_use:
        mascara_water |= ((mask & flag_to_index[m])!=0)
    mascara_water = (mascara_water == 1)

    mascara |= (~mascara_water)

    return mascara


def load_mask_C2RCC(image_path):
    """
    laod the water mask from S3 image

    :param image_path: path to .data/ dir
    :return:
    """
    mask_file = os.path.join(image_path,"c2rcc_flags.hdr")
    mask = envi.open(mask_file)[:,:,0]
    mask = mask[:,:,0]
    dim_file,_ = os.path.splitext(image_path)
    dim_file+=".dim"

    e = emt.parse(dim_file).getroot()

    gp = e.find("./Flag_Coding[@name='c2rcc_flags']")
    flag_to_index = dict([(f.find("Flag_Name").text,int(f.find("Flag_Index").text)) for f in gp.getchildren()])
    mascara = np.zeros(mask.shape,dtype=mask.dtype)

    masks_to_use = ["Cloud_risk","Rtosa_OOS","Rtosa_OOR","Rhow_OOS","Rhow_OOR"]
    for m in masks_to_use:
        mascara |= ((mask & flag_to_index[m])!=0)

    mascara = (mascara==1)

    gp = e.find("./Flag_Coding[@name='quality_flags']")
    flag_to_index = dict([(f.find("Flag_Name").text, int(f.find("Flag_Index").text)) for f in gp.getchildren()])
    # # mascara2 = np.zeros(mask.shape, dtype=mask.dtype)
    #
    masks_to_use = ["land"]
    mascara_water = np.zeros(mask.shape, dtype=mask.dtype)
    for m in masks_to_use:
        mascara_water |= ((mask & flag_to_index[m]) != 0)
    mascara_water = (mascara_water == 1)

    mascara |= (~mascara_water)

    return mascara


def predict_image(img,
                  mask,
                  model=None,
                  y_range=None,
                  y_mean=0, step=50000,
                  predict_function=None):
    """
    Given a 3d np.array img it applies the model to each pixel of the image.

    :param img:3d np.array
    :param mask: 2d np.array
    :param model: model to apply
    :param y_range: (min,max) tuple to trim the values of the output of the model
    :param y_mean:
    :param step: number of pixels to predict on batch
    :param predict_function: predict function instead of model.predict
    :return: 2d np.array with the predicted data
    """

    if predict_function is None:
        predict_function = lambda data: model.predict(data)

    img_flat = np.reshape(img,(img.shape[0]*img.shape[1],img.shape[2]))
    mask_flat = np.reshape(mask,(mask.shape[0]*mask.shape[1]))

    img_flat = img_flat[~mask_flat]

    if step is None:
        step = img_flat.shape[0]

    indices = range(0, img_flat.shape[0], step)

    preds = np.concatenate([predict_function(img_flat[i:(i + step)]) for i in indices],
                           axis=0)

    preds += y_mean

    if y_range is not None:
        preds = np.clip(preds, y_range[0], y_range[1])

    predictions = np.ndarray((mask_flat.shape[0],) + preds.shape[1:],
                             dtype=preds.dtype)

    predictions[~mask_flat] = preds
    predictions[mask_flat] = np.nan
    # predictions[mask_flat] = -999

    predictions = predictions.reshape((img.shape[0],img.shape[1]))

    return predictions


HDR_HEADER = {'band names': ['modeled_parameter'],
 'bands': '1',
 'byte order': '1',
 'data type': '4',
 'description': '',
 'file type': 'ENVI Standard',
 'interleave': 'bsq',
 'lines': '',
 'samples': '',
 'sensor type': 'Unknown'}

def write_envi(img,hdr_file):
    """
    save image as envi

    :param img:
    :param hdr_file:
    :return:
    """
    header = dict(HDR_HEADER)
    header["lines"] = str(img.shape[0])
    header["samples"] = str(img.shape[0]*img.shape[1])
    header["data type"] = envi.dtype_to_envi[img.dtype.char]
    header["description"] = "Imagen: "+os.path.basename(hdr_file)
    envi.save_image(hdr_file,
                    img,metadata=header,
                    dtype=img.dtype.char,
                    force=True,
                    interleave=header["interleave"],
                    byteorder=header['byte order'])
