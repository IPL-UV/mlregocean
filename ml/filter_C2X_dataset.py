_author__ = 'Ana'

import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from geoinfopy.geoinfo import gicharts
import scipy as sp
import seaborn as sns
from scipy import stats

####Extract subsets of the fully normalized Rrs data for OLCI bands files "HL_20151110_Rrs_fn_realOLCIbands.txt" (without
# inelastic scattering) and "HL_20150901_Rrs_fn_realOLCIbands.txt" (with elastic scattering). Filter first for data
# that will be used for validation only using "P_IDs_Not_used_for_NN_training.dat"####

path = '/media/ana/Nuevo vol/IPL/Databases/C2X/'
# infile = "HL_20151110_Rrs_fn_realOLCIbands.txt"
# P_ID = "P_IDs_Not_used_for_NN_training.txt"
#
# HL = pd.read_csv(path + infile, sep = '\t', header = 0, na_values = ['NaN'])
# # print(len(HL), HL.head)
#
# datval = pd.read_csv(path + P_ID, sep = '\t', header = 0, na_values = ['NaN'])
# # print(len(datval), datval.head)
#
# owts =['C2A', 'C2AX']#'Case1','C2S','C2SX'
# for owt in owts:
#     owtfile = pd.read_csv(path + 'HL_20150901_'+ owt + '_Input.txt', sep="\t", na_values='NaN')
#     # print(len(owtfile))
#
#     ##Use datval to discard rows in HL
#     keys = ['Run_ID']
#     i1 = HL.set_index(keys).index
#     i2 = datval.set_index(keys).index
#     HL_train = HL[~i1.isin(i2)]
#     # print(len(HL_train))
#     HL_test = HL[i1.isin(i2)]
#
#     ##Merge with 'HL_20150901_C2*_Input.txt'
#     HL_owt_train = pd.merge(owtfile, HL_train, on='Run_ID', how='inner')
#     # print(len(HL_owt_train), HL_owt_train.head)
#     HL_owt_train.to_csv(path+ 'HL_' +owt+'_train.txt', sep='\t')
#
#     HL_owt_test = pd.merge(owtfile, HL_test, on='Run_ID', how='inner')
#     # print(len(HL_owt_test), HL_owt_test)
#     HL_owt_test.to_csv(path + 'HL_' + owt + '_test.txt', sep='\t')

####concatanate results
infile1 = "HL_C2A_test.txt"
infile1= pd.read_csv(path + infile1, sep = '\t', header = 0, na_values = ['NaN'])
infile2 = "HL_C2AX_test.txt"
infile2 = pd.read_csv(path + infile2, sep = '\t', header = 0, na_values = ['NaN'])
# print(len(infile1))#, print(len(infile2)))
frames = [infile1, infile2]
infile = pd.concat(frames)
infile.to_csv(path + 'HL_C2A_total_test.txt', sep='\t')