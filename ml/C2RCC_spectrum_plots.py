_author__ = 'Ana'

from glob import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
#import geoinfopy.geoinfo.gicharts as gicharts
# from scipy import stats, rand # linspace, polyval, polyfit, sqrt, stats
import scipy as sp
# from pylab import plot, title, show , legend
import seaborn as sns
from scipy import stats
from sys import exit
import os

path = '/Users/Anita/Desktop/C2RCC_consistency/OLCI/'
sites = ['GustavDalenTower']#, 'Ieodo','Zeebrugge']
modes = ['rhow','rhown']

for mode in modes:
    for site in sites:
        sitePath = path + site

        file_1 = sitePath + '/C2RCC_v1/S3A_OL_1_EFR_MERMAID_C2RCC_v1_' + mode + '_all_' + site + '.txt'
        file_2 = sitePath + '/C2RCC_alt/S3A_OL_1_EFR_MERMAID_C2RCC_alt_' + mode + '_all_' + site + '.txt'

        ### Second merge and plot
        df1 = pd.read_csv(file_1, sep='\t', header=0, na_values=[''])
        df2 = pd.read_csv(file_2, sep='\t', header=0, na_values=[''])
        #print(len(df1), len(df2))

        ## merge only when there is the data in the two tables for the same position
        bigDf = df1.merge(df2, right_on="MID", left_on="MID", how='inner')
        bigDf = bigDf.dropna()
        MID = np.array(bigDf['MID'])

        #print(MID)

        #print(len(bigDf))
        #bigDf.to_csv(sitePath +'/Table for spectrum plots.txt', sep='\t')

        for spectra in range(len(bigDf)):
            #if site == 'AAOT' or site =='GustavDalenTower':
            data_v1 = pd.DataFrame(bigDf.iloc[:, [3, 5, 7, 9, 11, 13, 15]])
            data_alt = pd.DataFrame(bigDf.iloc[:, [19, 21, 23, 25, 27, 29, 31]])
            data_IS = pd.DataFrame(bigDf.iloc[:, [2, 4, 6, 8, 10, 12, 14]])

            #elif site == 'Ieodo' or site =='Zeebrugge':
            #    data_v1 = pd.DataFrame(bigDf.iloc[:, [3, 5, 7, 9, 11, 13]])
            #    data_alt = pd.DataFrame(bigDf.iloc[:, [17, 19, 21, 23, 25, 27]])
            #    data_IS = pd.DataFrame(bigDf.iloc[:, [2, 4, 6, 8, 10, 12]])


            data_v1 = data_v1.transpose()
            data_alt = data_alt.transpose()
            data_IS = data_IS.transpose()
            bands = [410, 440, 490, 510, 560, 665, 865]
            data_IS['Bands'] = bands
            data_v1['Bands'] = bands
            data_alt['Bands'] = bands
            #print(data_IS)
            #print(data_v1)
            wavelengths= pd.DataFrame([400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530,
                                       540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 665,
                                       670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800,
                                       810, 820, 830, 840, 850, 860, 865,870])
            wavelengths.rename(columns={0: 'Bands'}, inplace=True)
            #print(wavelengths)
            data_IS_plot = pd.merge(data_IS, wavelengths, how ='right', on = 'Bands')
            data_IS_plot.sort_values('Bands', axis=0, ascending=True, inplace=True)
            data_v1_plot = pd.merge(data_v1, wavelengths, how='right', on='Bands')
            data_v1_plot.sort_values('Bands', axis=0, ascending=True, inplace=True)
            data_alt_plot = pd.merge(data_alt, wavelengths, how='right', on='Bands')
            data_alt_plot.sort_values('Bands', axis=0, ascending=True, inplace=True)

            print(data_v1_plot)

            MIDnum = str(MID[spectra])
            # print(MIDnum)
            #fig = plt.figure()
            ax = data_v1_plot.plot(x=data_IS_plot.Bands, y=data_v1_plot.columns[spectra], legend=False, grid=True, color='blue',
                              marker='*',markersize=10, label='v1',linestyle='-')
            data_alt_plot.plot(x=data_IS_plot.Bands, y=data_alt_plot.columns[spectra], legend=False, grid=True, color='red',
                          marker='+',markersize=10, label='alt',linestyle='-', ax=ax)
            data_IS_plot.plot(x=data_IS_plot.Bands, y=data_IS_plot.columns[spectra], legend=False, grid=True,
                         color='black', marker='.',markersize=10, label='IS',linestyle='-', ax=ax)
            ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance ($sr^1$)")


            if site == 'AAOT':
                ax.set_title('Spectrum shape ' +mode+' '+ site + ' VICAL (' + MIDnum + ')' , size=14)
            else:
                ax.set_title('Spectrum shape ' +mode+' '+ site + ' (' + MIDnum + ')', size=14)

            '''
            if site == 'AAOT' or site == 'GustavDalenTower':
                wv = ['412.5', '442.5', '490', '510','560', '665', '865']
            elif site == 'Ieodo' or site == 'Zeebrugge':
                wv = ['412.5', '442.5', '490', '560', '665', '865']
            '''


            #ax.set_xticklabels(wv)
            ax.grid(color='lightgrey', linestyle='--')
            ax.set_ylim(-0.005, 0.1)
            plt.legend()


            #plt.show()
            plt.savefig(sitePath+'/Spectrum_plot_'+site+'_'+mode+'_'+MIDnum+'_v2.png')
            plt.close()
            #break