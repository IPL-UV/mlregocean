__author__ = 'Gonzalo Mateo, Ana Ruescas'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error ,mean_absolute_error ,r2_score
import warnings
import seaborn as sns
import scipy.stats
import time
import scipy as sp


warnings.filterwarnings("ignore")

def get_rmse_abs(preds, measured):
    """Root mean squared error:

    :param x_values: x values
    :param y_values: y values
    :return: RMSE
    """

    # Source: errorlib.py (http://civil.iisc.ernet.in/~satkumar/)
    squared_differences = (preds - measured) ** 2
    rmse = np.sqrt(np.mean(squared_differences))

    return rmse


def get_rmse_rel(preds, measured):
        return np.sqrt(np.mean(((preds - measured) / measured) ** 2))


def get_bias(preds, measured):
    """ Calculate the absolute bias of a list of x and y values:

    :param x_values:
    :param y_values:
    :return: Bias
    """

    # Source: errorlib.py (http://civil.iisc.ernet.in/~satkumar/)
    if len(preds) != 0:
        bias = np.sum(preds - measured) / len(measured)
    else:
        bias = np.nan

    return bias


fd = os.open('/media/ana/Nuevo vol/IPL/Databases/C2X/ValidationProducts/',os.O_RDONLY)
os.fchdir(fd)
print(os.getcwd() + "\n")

skdata1= pd.read_csv('ONNSv04_CDOM_vs_C2X_validation_data_C2A.dat', sep='\t', na_values=' ')
skdata2= pd.read_csv('ONNSv04_CDOM_vs_C2X_validation_data_C2AX.dat', sep='\t', na_values=' ')
#print(len(skdata1), len(skdata2))

skdata = pd.concat([skdata1,skdata2], axis=0)
#print(len(skdata))
y_test = skdata['a_440_cdom_ONNS']
preds = skdata['a_440_cdom_Validation']

mask = ~np.isnan(y_test) & ~np.isnan(preds)
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(y_test[mask], preds[mask])

a = str("{0:.5f}".format(slope))
b = str("{0:.5f}".format(intercept))
c = str("{0:.5f}".format(r_value))
d = str("{0:.1E}".format(std_err))

stats = pd.DataFrame(
    {'Slope': [a], 'Intercept': [b], 'r': c, 'error': [d]})
print(stats)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(np.log10(y_test), np.log10(preds))
plt.ylabel("CDOM predicted", fontsize=16)
plt.xlabel("CDOM measured", fontsize=16)
ax.xaxis.grid(color='lightgrey', linestyle='dashed')
ax.yaxis.grid(color='lightgrey', linestyle='dashed')
ax.patch.set_facecolor('white')
ax.set_title("Validation C2AX ONNS CDOM", fontsize=25)
# ax.set_xlim(-5, 30)
# ax.set_ylim(-5, 30)
plt.xticks(fontsize=15, rotation=0)
plt.yticks(fontsize=15, rotation=0)
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="lightgrey")
fig.text(0.2, 0.8, r'$y=$' + a + r'$x$' + b, ha='left', fontsize=15)
fig.text(0.2, 0.76, r'$r=$' + c, ha='left', fontsize=15)
fig.text(0.2, 0.72, r'$error=$' + d, ha='left', fontsize=15)
plt.show()
#fig.savefig("VALIDATION_C2AX_ONNS_CDOM_log10.pdf")

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(residuals, preds)
# ax.xaxis.grid(color='lightgrey', linestyle='dashed')
# ax.yaxis.grid(color='lightgrey', linestyle='dashed')
# ax.patch.set_facecolor('white')
# # ax.set_xlim(-10, 20)
# # ax.set_ylim(-10, 20)
# plt.ylabel("Residuals", fontsize=16)
# plt.xlabel("CDOM predicted", fontsize=16)
# plt.xticks(fontsize=15, rotation=0)
# plt.yticks(fontsize=15, rotation=0)
# ax.set_title("Residuals_" + name + "_" + name_bands, fontsize=25)
# # plt.show()
# fig.savefig("RESIDUALS_" +  name + "_" + name_bands + "_CDOM.pdf")