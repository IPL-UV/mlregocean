
import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV ,KFold
from sklearn.linear_model import Ridge ,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error ,mean_absolute_error ,r2_score
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures ,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from glob import glob
import os
import spectral.io.envi as envi
import xml.etree.ElementTree as emt
from scipy import sparse

# bands_S3_syke = ['S3400','S3412.5','S3442.5','S3490','S3510','S3560',
#                   'S3620','S3665','S3673.75','S3681.25','S3708.75', 'S3753.75']
# bands_S3ratio = ['S3ratio1','S3ratio2']
# bands_S3_plus_ratios = bands_S3_syke + bands_S3ratio
#
# path = '/media/ana/Nuevo vol/IPL/Databases/'
#
# relation_band_names = {
#     "01": "S3400",
#     "02": 'S3412.5',
#      "03": 'S3442.5',
#      "04": 'S3490',
#      "05": 'S3510',
#      "06":'S3560',
#      "07":'S3620',
#      "08":'S3665',
#      "09":'S3673.75',
#      "10":'S3681.25',
#      "11":'S3708.75',
#      "12":'S3753.75',
#      "S3ratio1" : 'S3ratio1',
#      "S3ratio2" : 'S3ratio2'
#      }
#
# def read_data_syke(file=path + 'SYKE/SYKE_5553_Run2_out_S2_S3_header.txt'):
#     skdata = pd .read_csv(file,
#                       sep='\t', na_values=' ')
#     skdata['S2ratio1'] = skdata['S2665']/skdata['S2490']
#     skdata['S2ratio2'] = skdata['S2705']/skdata['S2490']
#     skdata['S3ratio1'] = skdata['S3665']/skdata['S3490']
#     skdata['S3ratio2'] = skdata['S3708.75']/skdata['S3490']
#
#     ###2.1 Other possibile inputs
#     bands_S2 = ['S2443','S2490','S2560','S2665','S2705','S2740']
#     bands_S2ratio = ['S2ratio1','S2ratio2']
#     bands_S3ratio = ['S3ratio1','S3ratio2']
#     bands_S2_plus_ratios = bands_S2 + bands_S2ratio
#     bands_S3_plus_ratios = bands_S3_syke + bands_S3ratio
#
#     #print(skdata.columns)
#     return skdata

### C2X dataset
bands_S3 = ['400', '412.5', '442.5', '490', '510', '560',
            '620', '665','673.75','681.25', '708.75','753.75', '778.75','865',
            '885']
bands_S3ratio = ['S3ratio1','S3ratio2']
bands_S3_plus_ratios = bands_S3 + bands_S3ratio

path = '/media/ana/Nuevo vol/IPL/Databases/'

relation_band_names = {
     "01": "400",
     "02": '412.5',
     "03": '442.5',
     "04": '490',
     "05": '510',
     "06":'560',
     "07":'620',
     "08":'665',
     "09":'673.75',
     "10":'681.25',
     "11":'708.75',
     "12":'753.75',
     "16":"778.75",
     "17": "865",
     "18": "885",
     "S3ratio1" : 'S3ratio1',
     "S3ratio2" : 'S3ratio2'
     }

def read_data_syke(file=path + 'C2X/HL_C2A_total_test.txt'):
    skdata = pd .read_csv(file,
                      sep='\t', na_values=' ')
    skdata = skdata[skdata['Chl_comp_1'] == 1]
    skdata['S3ratio1'] = skdata['665']/skdata['490']
    skdata['S3ratio2'] = skdata['708.75']/skdata['490']

    ###2.1 Other possibile inputs
    # bands_S3ratio = ['S3ratio1', 'S3ratio2']
    bands_S3_plus_ratios = bands_S3 + bands_S3ratio

    #print(skdata.columns)
    return skdata


def load_model(skdata,model,bands):
    X = skdata[bands]
    # cdom_array = np.asarray(skdata['a400 (1/m)'])
    cdom_array = np.asarray(skdata['a_440_cdom'])

    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    ###4. Split training-testing data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, cdom_array, test_size=0.25, random_state=42)

    mean_y = np.mean(y_train)
    y_train_norm = y_train - mean_y

    model.fit(X_train,y_train_norm)

    return min_max_scaler,mean_y

def load_S3_image(image_path= path + "subset_0_of_S3A_OL_2_WFR____20170708T084007_20170708T084307_20170708T104714_0179_019_335_1979_MAR_O_NR_002.data/"):
    inputs = [ip for ip in sorted(glob(os.path.join(image_path,"Oa*.hdr"))) if "err" not in ip]

    images = []
    band_names = []
    for ip in inputs:
        band_name = os.path.splitext(os.path.basename(ip))[0]
        band_name = band_name.replace("_reflectance","").replace("Oa","")
        band_names.append(band_name)
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
    band_names.append("S3ratio1")
    images.append(S3ratio2)
    band_names.append("S3ratio2")
    images = np.concatenate(images,axis=2)




    return images, band_names



def load_mask(image_path= path + "subset_0_of_S3A_OL_2_WFR____20170708T084007_20170708T084307_20170708T104714_0179_019_335_1979_MAR_O_NR_002.data/"):
    mask_file = os.path.join(image_path,"WQSF_lsb.hdr")
    mask = envi.open(mask_file)[:,:,0]
    mask = mask[:,:,0]
    e = emt.parse(path + "subset_0_of_S3A_OL_2_WFR____20170708T084007_20170708T084307_20170708T104714_0179_019_335_1979_MAR_O_NR_002.dim").getroot()

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


def predict_image(img, band_names, mascara,model, scaler, bands_model, y_range=None, y_mean=0, step=50000,predict_function=None):

    if predict_function is None:
        predict_function = lambda data: model.predict(data)

    img_flat = np.reshape(img,(img.shape[0]*img.shape[1],img.shape[2]))
    mask_flat = np.reshape(mascara,(mascara.shape[0]*mascara.shape[1]))

    img_flat = pd.DataFrame(img_flat,columns=band_names)
    img_flat = img_flat.rename(columns=relation_band_names)

    img_to_predict = img_flat[bands_model].as_matrix()
    # img_to_predict = img_flat[bands_S3_plus_ratios]

    img_flat = img_to_predict[~mask_flat]

    if step is None:
        preds = predict_function(scaler.transform(img_flat))
    else:
        indices = range(0, img_flat.shape[0], step)
        preds = np.concatenate([predict_function(scaler.transform(img_flat[i:(i + step)])) for i in indices],
                           axis=0)

    preds += y_mean
    if y_range is not None:
        preds = np.clip(preds, y_range[0], y_range[1])

    predictions = np.ndarray((mask_flat.shape[0],) + preds.shape[1:],
                             dtype=preds.dtype)

    predictions[~mask_flat] = preds
    predictions[mask_flat] = np.nan

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
