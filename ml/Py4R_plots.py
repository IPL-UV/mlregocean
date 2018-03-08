__author__ = 'Ana Ruescas, Gonzalo Mateo-García'

""" Plots that can be done with the result of the Py4R ML regression methods:
    1. boxplots of the spectrum for summary of info
    2. correlative plot: scatter plots + annotations
    3. partial plots
    4. permutation plots
    5. boxplots of error by model   
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy as sp
from sklearn.metrics import mean_absolute_error


def correlative_plot(preds, measured,ax=None,s=10):
    """
    Correlative plot
    Correlative plot prediction vs. measured: as validation plot using linear regression;
    (also useful for residual vs. measured)
    Inputs:  and measured (y_test) data
    Results from models are stored in:
     predictions = pd.DataFrame(predictions).T
     predictions.to_csv(name_bands + "_preds.csv", index=False)
    :param preds: two arrays with predictions (preds)
    :param measured:
    :param ax:
    :param s:
    :return:
    """

    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(measured, preds)

    a = str("{0:.5f}".format(slope))
    b = str("{0:.5f}".format(intercept))
    c = str("{0:.5f}".format(r_value))
    d = str("{0:.1E}".format(std_err))

    if ax is None:
        ax = plt.gca()

    ax.scatter(measured, preds,s=s)
    ax.set_ylabel( "predicted")
    ax.set_xlabel( "measured")
    ax.xaxis.grid(color='lightgrey', linestyle='dashed')
    ax.yaxis.grid(color='lightgrey', linestyle='dashed')
    ax.patch.set_facecolor('white')
    #ax.set_title("Validation", fontsize=25) #example
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="lightgrey")

    ## Annotate plot with regression formula -->
    ax.text(0.2, np.max(preds)-1, r'$y=$' + a + r'$x$' + b, ha='left', fontsize=15)
    ax.text(0.2, np.max(preds)-5, r'$r=$' + c, ha='left', fontsize=15)
    ax.text(0.2, np.max(preds)-9, r'$error=$' + d, ha='left', fontsize=15)


def regression_plot(x_train, x_test, y_train, y_test, models_predict, mean_y_train=0, ax=None):
    if ax is None:
        ax = plt.gca()

    rango_x = np.linspace(np.min((np.min(x_test), np.min(x_train))),
                          np.max((np.max(x_test), np.max(x_train))), 200)
    i = 0
    for model_name, model in models_predict:
        if model_name == "GPR":
            min_max_scaler_ = model.steps[0][1]
            gpr_model_ = model.steps[1][1]
            yp, y_std = gpr_model_.predict(min_max_scaler_.transform(rango_x[:, np.newaxis]),
                                           return_std=True)
            yp += mean_y_train
            ax.fill_between(rango_x, yp - 1.96 * y_std, yp + 1.96 * y_std,
                            alpha=0.2, color="C%d" % i)
        else:
            yp = model.predict(rango_x[:, np.newaxis])
            yp += mean_y_train
        ax.plot(rango_x, yp, label=model_name)
        i += 1

    ax.scatter(x_test, y_test, s=8, label="test", alpha=.5,c="C%d"%i)
    ax.scatter(x_train, y_train, s=8, label="train", alpha=.5,c="C%d"%(i+1))


def permutation_test(model,X,y, function_error=mean_absolute_error,P=30):
    salida = pd.DataFrame(columns=X.columns, data=np.ndarray((P, X.shape[1])))
    # print("Error inicial: %.3f"%(function_error(model.predict(X),y)))
    for col in X.columns:
        error = np.ndarray((P,))
        X_copia = X.copy()
        for i in range(P):
            x_datos = np.array(X_copia[col])
            x_datos = x_datos[np.random.permutation(X.shape[0])]
            X_copia[col] = x_datos
            y_pred = model.predict(X_copia)
            error[i] = function_error(y, y_pred)
        salida[col] = error
    return pd.melt(salida,value_name=function_error.__name__)









"""

'''The first plot is very much fitted to RS datasets'''
## Boxplots by band for whole spectrum: this is just an statistical summary of the input bands
## Inputs: df with bands per pixel
def boxplots_bands(df):

    skdata = pd.read_csv('SYKE_5553_Run2_out_S2_S3_header.txt', sep='\t', na_values=' ')
    bands_S2 = ['S2443', 'S2490', 'S2560', 'S2665', 'S2705', 'S2740']
    bands_S3 = ['S3400', 'S3412.5', 'S3442.5', 'S3490', 'S3510', 'S3560',
                'S3620', 'S3665', 'S3673.75', 'S3681.25', 'S3708.75', 'S3753.75']
    dataplots2 = skdata[['S2443', 'S2490', 'S2560', 'S2665', 'S2705', 'S2740']]
    dataplots3 = skdata[['S3400', 'S3412.5', 'S3442.5', 'S3490', 'S3510', 'S3560',
                         'S3620', 'S3665', 'S3673.75', 'S3681.25', 'S3708.75', 'S3753.75']]
    dataplots = [dataplots2, dataplots3]

    for data in dataplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.boxplot(data.values)
        ax.set_ylim(-0.001, 0.01)
        if data is dataplots[0]:
            ax.set_xticklabels(bands_S2, ) ## maybe there is another way of doing this
        elif data is dataplots[1]:
            ax.set_xticklabels(bands_S3)
        ax.set_title("Spectral bands statistics", fontsize=15)
        ax.xaxis.grid(color='lightgrey', linestyle='dashed')
        ax.yaxis.grid(color='lightgrey', linestyle='dashed')
        ax.patch.set_facecolor('white')
        # ax.set_xlabel("Spectral bands", fontsize=12)
        plt.xticks(fontsize=12, rotation=90)
        plt.yticks(fontsize=12, rotation=0)
        if data is dataplots[0]: #example title, probably other ways possible
            fig.savefig('Spectral_bands_statistics_Sentinel2.pdf')
        elif data is dataplots[1]:
            fig.savefig('Spectral_bands_statistics_Sentinel3.pdf')

def partial_plots(model ,X):
    P = 20
    PPLOTS = pd.DataFrame(columns=X.columns, data=np.ndarray((P, X.shape[1])))
    XPLOTS = pd.DataFrame(columns=X.columns, data=np.ndarray((P, X. shape[1])))
    for col in X.columns:
        min_ = np.min(X[col])
        max_ = np.max(X[col])
        xc = np.linspace(min_,max_ ,P)
        XPLOTS[col] = xc

        y_pred_media = np.ndarray((len(xc,)))
        for i,x in enumerate(xc):
            X_copia = X.copy()
            X_copia[col] = x
            y_pred_media[i] = np.mean(model.predict(X_copia))

        PPLOTS[col] = y_pred_media

    return XPLOTS, PPLOTS

    #Plotting
    fig, ax = plt.subplots(4, 3) #This can change depending on input bands
    ##MSI
    # fig, ax = plt.subplots(3, 2)
    ax = ax.flatten()


    for axe, col in zip(ax, XPLOTS.columns):
        axe.plot(XPLOTS[col], PPLOTS[col])
        axe.scatter(X_test_df[col], y_test, c="grey", marker="+", s=10)
        axe.tick_params(axis='both', which='major', labelsize=10)
        axe.xaxis.grid(color='lightgrey', linestyle='dotted')
        axe.yaxis.grid(color='lightgrey', linestyle='dotted')
        axe.patch.set_facecolor('white')
        plt.xticks(rotation='vertical')
        plt.subplots_adjust(bottom=0.25)

    #TODO: read name of bands and select all automatically for subplotting (it usually changes by sensor)
    ax[0].set_title('400 nm', fontsize=10)
    ax[1].set_title('412 nm', fontsize=10)
    ax[2].set_title('443 nm', fontsize=10)
    ax[3].set_title('490 nm', fontsize=10)
    ax[4].set_title('510 nm', fontsize=10)
    ax[5].set_title('560 nm', fontsize=10)
    ax[6].set_title('620 nm', fontsize=10)
    ax[7].set_title('665 nm', fontsize=10)
    ax[8].set_title('673 nm', fontsize=10)
    ax[9].set_title('753 nm', fontsize=10)
    ax[10].set_title('778 nm', fontsize=10)
    ax[11].set_title('865 nm', fontsize=10)

    # ax[9].set_title('681 nm', fontsize=10)
    # ax[10].set_title('708 nm', fontsize=10)
    # ax[11].set_title('753 nm', fontsize=10)

    fig.subplots_adjust(hspace=0.4)
    fig.suptitle(name_bands + "_" + name, fontsize=10)


    # plt.savefig("partial_plots_"+name_bands+"_"+name+".pdf")


# Permutation plots with RMSE of bands as inputs (per model)
## Inputs: model, X_test_scaled_df, y_test, mean_y
## I do not know how to do this, maybe with .pkl file??


    ##Plotting

## Linear regression plots of one variable (ratio) per model
## Inputs:
def regression_models():

    if len(bands) == 1:
        rango_x = np.linspace(np.min(X_test[:, 0]), np.max(X_test[:, 0]), 200)
        #rango_x = np.linspace(np.min(X_test_scaled[:, 0]), np.max(X_test_scaled[:, 0]), 200)
        fig, ax = plt.subplots(1, 1)
        # ax = ax.flatten()
        plt.scatter(X_test[:, 0], y_test_norm)
        # plt.scatter(X_test[:, 0], y_test)
        plt.scatter(X_train[:, 0], y_train_norm, c="green")
        for name, model in models:
            if name == "GPR":
                print("Adding error bars")
                yp, y_std = model.predict(min_max_scaler.transform(rango_x[:, np.newaxis]), return_std=True)
                plt.fill_between(rango_x, yp - 1.96 * y_std, yp + 1.96 * y_std,
                                 alpha=0.2, color="darkorange")
            else:
                yp = model.predict(min_max_scaler.transform(rango_x[:, np.newaxis]))
            plt.plot(rango_x, yp, label=name)

        plt.legend(loc=4)
        # plt.show()

        ax.set_title("Regression method comparison with " + name_bands, fontsize=20)
        ax.xaxis.grid(color='lightgrey', linestyle='dashed')
        ax.yaxis.grid(color='lightgrey', linestyle='dashed')
        ax.patch.set_facecolor('white')
        # ax.set_xlim(0.,1.)
        # ax.set_ylim(0, 60)
        fig.savefig("regression_" + name_bands + ".pdf")

## Boxplots of errors by model in one chart
## Inputs:
def boxplot_models():

    fig = plt.figure()
    # fig.suptitle('Algorithm Comparison MAE')
    ax = fig.add_subplot(111)
    plt.boxplot(results_mae)
    ax.set_ylim(-0.5, 5)
    ax.set_xticklabels(names)
    ax.set_title("MAE_" + name_bands, fontsize = 25)
    ax.xaxis.grid(color='lightgrey', linestyle='dashed')
    ax.yaxis.grid(color='lightgrey', linestyle='dashed')
    ax.patch.set_facecolor('white')
    plt.xlabel("MODELS", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    fig.savefig("MAE_" + name_bands + ".pdf")

    # fig = plt.figure()
    # # fig.suptitle('Algorithm Comparison residuals')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results_residuals)
    # ax.set_ylim(-10, 10)
    # ax.set_xticklabels(names)
    # ax.set_title("ERRORS_" + name_bands, fontsize = 25)
    # ax.xaxis.grid(color='lightgrey', linestyle='dashed')
    # ax.yaxis.grid(color='lightgrey', linestyle='dashed')
    # ax.patch.set_facecolor('white')
    # plt.xlabel("MODELS", fontsize=16)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # fig.savefig("ERRORS_" + name_bands + ".pdf")

"""