__author__ = 'Gonzalo Mateo-García, Ana Ruescas'

import pandas as pd
from sklearn.model_selection import train_test_split

PATH_TO_DATA_SYKE = "/media/disk/databases/SYKE/SYKE_5553_Run2_out_S2_S3_header.txt"
PATH_TO_DATA_TRAIN_C2X = "/media/disk/databases/C2X/HL_C2A_total_train.txt"
PATH_TO_DATA_TEST_C2X = "/media/disk/databases/C2X/HL_C2A_total_test.txt"
PATH_TO_MODELS_SYKE = "/media/disk/databases/SYKE/models"
PATH_TO_MODELS_C2X = "/media/disk/databases/C2X/models"

###2.1 Possible inputs
bands_S2_SYKE = ['S2443','S2490','S2560','S2665','S2705','S2740']
bands_S3_SYKE = ['S3400','S3412.5','S3442.5','S3490','S3510','S3560',
                  'S3620','S3665','S3673.75','S3681.25',
                 'S3708.75', 'S3753.75']
bands_S2ratio = ['S2ratio1','S2ratio2']
bands_S3ratio = ['S3ratio1','S3ratio2']
bands_S2_plus_ratios_SYKE = bands_S2_SYKE + bands_S2ratio
bands_S3_plus_ratios_SYKE = bands_S3_SYKE + bands_S3ratio

ALL_BANDS_SYKE = bands_S2_plus_ratios_SYKE + bands_S3_plus_ratios_SYKE
target_band_SYKE = 'a400 (1/m)'

bands_try_SYKE=dict([("S2bands",bands_S2_SYKE),
           ('S2ratio1', ['S2ratio1']),
           ('S2ratio2', ['S2ratio2']),
           ("ratios_S2", bands_S2ratio),
           ("S2bands&ratios", bands_S2_plus_ratios_SYKE),
           ('S3ratio1',['S3ratio1']),
           ('S3ratio2',['S3ratio2']),
           ("ratios_S3",bands_S3ratio),
           ("S3bands",bands_S3_SYKE),
           ("S3bands&ratios", bands_S3_plus_ratios_SYKE)])

def load_SYKE(path_to_data=PATH_TO_DATA_SYKE):
    skdata = pd.read_csv(path_to_data, sep='\t', na_values=' ')
    skdata['S2ratio1'] = skdata['S2665'] / skdata['S2490']
    skdata['S2ratio2'] = skdata['S2705'] / skdata['S2490']
    skdata['S3ratio1'] = skdata['S3665'] / skdata['S3490']
    skdata['S3ratio2'] = skdata['S3708.75'] / skdata['S3490']

    skdata_X_train, skdata_X_test, skdata_y_train, skdata_y_test = train_test_split(skdata[ALL_BANDS_SYKE],
                                                                                    skdata[target_band_SYKE], test_size=0.25, random_state=42)
    skdata_y_train = skdata_y_train.values
    skdata_y_test = skdata_y_test.values

    return skdata_X_train, skdata_X_test, skdata_y_train, skdata_y_test


bands_S3_C2X = ['400', '412.5', '442.5', '490', '510', '560',
                '620', '665','673.75','681.25', '708.75','753.75',
                '778.75','865',
                '885']
bands_S3_plus_ratios_C2X = bands_S3_C2X + bands_S3ratio

bands_try_C2X=dict([("S3bands", bands_S3_C2X),
                ('S3ratio1',['S3ratio1']),
                ('S3ratio2',['S3ratio2']),
                ('S3bands&ratios', bands_S3_plus_ratios_C2X),
                ("ratios_S3",bands_S3ratio)])

ALL_BANDS_C2X = bands_S3_plus_ratios_C2X
CDOM_C2X_variable_name = 'a_440_cdom'
TSM_C2X_variable_name = 'TSM'
CHL_C2X_variable_name = 'Chl'
TARGET_VARIABLES_C2X = [CDOM_C2X_variable_name,TSM_C2X_variable_name,CHL_C2X_variable_name]
TARGET_VARIABLES_C2X_NAMED = dict(zip(["CDOM","TSM","Chl"],TARGET_VARIABLES_C2X))

def load_C2X(path_to_train=PATH_TO_DATA_TRAIN_C2X,
             path_to_test=PATH_TO_DATA_TEST_C2X,
             target_variables=TARGET_VARIABLES_C2X):

    skdata = pd.read_csv(path_to_train, sep='\t', na_values=' ')
    skdata = skdata[skdata['Chl_comp_1'] == 1]

    skdata['S3ratio1'] = skdata['665'] / skdata['490']
    skdata['S3ratio2'] = skdata['708.75'] / skdata['490']


    skdata_test = pd.read_csv(path_to_test,
                              sep='\t', na_values=' ')

    skdata_test['S3ratio1'] = skdata_test['665'] / skdata_test['490']
    skdata_test['S3ratio2'] = skdata_test['708.75'] / skdata_test['490']

    skdata_test = skdata_test[skdata_test['Chl_comp_1'] == 1]

    skdata_y_train = skdata[target_variables]
    skdata_y_test = skdata_test[target_variables]

    return skdata[ALL_BANDS_C2X], skdata_test[ALL_BANDS_C2X], skdata_y_train,skdata_y_test




