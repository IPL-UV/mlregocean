
__author__ = 'Ana Ruescas'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import scipy.stats
import time
import scipy as sp
#from matplotlib.axes import Subplot as plt

warnings.filterwarnings("ignore")

fd = os.open('/media/ana/Nuevo vol/IPL/Databases/SYKE/', os.O_RDONLY)
os.fchdir(fd)
print(os.getcwd() + "\n")

###1. Read dataset
skdata= pd.read_csv('SYKE_5553_Run2_out_S2_S3_header.txt', sep='\t', na_values=' ')
bands_S2 = ['S2443','S2490','S2560','S2665','S2705','S2740']
bands_S3 = ['S3400','S3412.5','S3442.5','S3490','S3510','S3560',
                  'S3620','S3665','S3673.75','S3681.25','S3708.75', 'S3753.75']


# sns.boxplot(bands_S2, data=skdata)
dataplots2=skdata[['S2443','S2490','S2560','S2665','S2705','S2740']]
dataplots3=skdata[['S3400','S3412.5','S3442.5','S3490','S3510','S3560',
                  'S3620','S3665','S3673.75','S3681.25','S3708.75', 'S3753.75']]
dataplots = [dataplots2,dataplots3]
'''
for data in dataplots:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.boxplot(data.values)
    ax.set_ylim(-0.001, 0.01)
    if data is dataplots[0]:
        ax.set_xticklabels(bands_S2, )
    elif data is dataplots[1]:
        ax.set_xticklabels(bands_S3)
    ax.set_title("Spectral bands statistics", fontsize=15)
    ax.xaxis.grid(color='lightgrey', linestyle='dashed')
    ax.yaxis.grid(color='lightgrey', linestyle='dashed')
    ax.patch.set_facecolor('white')
    # ax.set_xlabel("Spectral bands", fontsize=12)
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12, rotation=0)
    # if data is dataplots[0]:
    #     fig.savefig('Spectral_bands_statistics_Sentinel2.pdf')
    # elif data is dataplots[1]:
    #     fig.savefig('Spectral_bands_statistics_Sentinel3.pdf')
'''
wavelengths = pd.DataFrame([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530,
                            540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 665,
                            670, 680, 690, 700, 710, 720, 730, 740, 750, 760])
wavelengths.rename(columns={0: 'Bands'}, inplace=True)
bands_s2 = [440, 490, 560, 665, 700, 740]
bands_s3 = [400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.5, 681.5, 708.75, 753.75]

for data in dataplots:


    ###TODO: merge on data before calculating max, min, mean

    meanline = pd.DataFrame(data.mean())
    meanline['Bands'] = meanline.index
    # print(meanline)
    if data is dataplots2:
        meanline['Bands'] = bands_s2
    elif data is dataplots3:
        meanline['Bands'] = bands_s3
    meanline = pd.merge(meanline, wavelengths, how='right', on='Bands')
    meanline.rename(columns={0: 'Values'}, inplace=True)
    meanline.sort_values('Bands', axis=0, ascending=True, inplace=True)
    newmean = meanline.interpolate(method='linear', axis=0).ffill().bfill()
    # print(newmean)


    maxline = pd.DataFrame(data.max())
    maxline['Bands'] = maxline.index
    if data is dataplots2:
        maxline['Bands'] = bands_s2
    elif data is dataplots3:
        maxline['Bands'] = bands_s3
    maxline = pd.merge(maxline, wavelengths, how='right', on='Bands')
    maxline.rename(columns={0: 'Values'}, inplace=True)
    maxline.sort_values('Bands', axis=0, ascending=True, inplace=True)
    newmax = maxline.interpolate(method='linear', axis=0).ffill().bfill()


    minline = pd.DataFrame(data.min())
    minline['Bands'] = minline.index
    if data is dataplots2:
        minline['Bands'] = bands_s2
    elif data is dataplots3:
        minline['Bands'] = bands_s3
    minline = pd.merge(minline, wavelengths, how='right', on='Bands')
    minline.rename(columns={0: 'Values'}, inplace=True)
    minline.sort_values('Bands', axis=0, ascending=True, inplace=True)
    newmin = minline.interpolate(method='linear', axis=0).ffill().bfill()
    # print(newmin)


    ax = newmean.plot(x=newmean.Bands, y=newmean.columns[0], legend=False, grid=True,
                      color='darkred', linestyle='-', label='mean')
    newmax.plot(x=newmean.Bands, y=newmax.columns[0], legend=False, grid=True,
                color='lightgrey',linestyle='-', ax=ax, label='max')
    newmin.plot(x=newmean.Bands, y=newmin.columns[0], legend=False, grid=True,
                color='grey',linestyle='-',ax=ax, label='min')
    ax.set_title("Spectral shape summary", fontsize=15)
    ax.fill_between(newmean.Bands, newmean.Values, newmax.Values,facecolor='whitesmoke')
    ax.fill_between(newmean.Bands, newmean.Values, newmin.Values, facecolor='whitesmoke')
    ax.xaxis.grid(color='lightgrey', linestyle='dashed')
    ax.yaxis.grid(color='lightgrey', linestyle='dashed')
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12, rotation=0)
    ax.patch.set_facecolor('white')
    ax.set_ylim(-0.005, 0.03)

    plt.legend()

    plt.show()
    # if data is dataplots[0]:
    #     plt.savefig('Spectral_shape_Sentinel2.pdf')
    # elif data is dataplots[1]:
    #     plt.savefig('Spectral_shape_Sentinel3.pdf')
    break
