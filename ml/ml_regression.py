
__author__ = 'Gonzalo Mateo, Ana Ruescas'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import warnings
from sklearn.externals import joblib
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

# def partial_plots(model ,X):
#     P = 20
#     PPLOTS = pd.DataFrame(columns=X.columns, data=np.ndarray((P, X.shape[1]))) XPLOTS=pd. DataFrame(columns=X.columns, data=np.ndarray((P,X. shape[1])))
#     for col in X.columns:
#         min_ = np.min(X[col])
#         max_ = np.max(X[col])
#         xc = np.linspace(min_,max_ ,P)
#         XPLOTS[col] = xc
#
#         y_pred_media = np.ndarray((len(xc,)))
#         for i,x in enumerate(xc):
#             X_copia = X.copy()
#             X_copia[col] = x
#             y_pred_media[i] = np.mean(model.predict(X_copia))
#
#         PPLOTS[col] = y_pred_media
#
#     return XPLOTS, PPLOTS

def permutation_test(model,X,y , function_error=mean_absolute_error):
    P = 30
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
    return salida


# fd = os.open('/media/ana/Nuevo vol/IPL/Databases/SYKE/', os.O_RDONLY)
fd = os.open('/media/ana/Nuevo vol/IPL/Databases/C2X/TSM/',os.O_RDONLY)
os.fchdir(fd)
print(os.getcwd() + "\n")

###1. Read dataset
# skdata= pd .read_csv('SYKE_5553_Run2_out_S2_S3_header.txt', sep='\t', na_values=' ')
skdata= pd .read_csv('HL_C2A_total_train.txt', sep='\t', na_values=' ')
skdata = skdata[skdata['Chl_comp_1']==1]

##2. Calculate ratios

##C2X dataset
# skdata['S3ratio1'] = skdata['665']/ skdata['490']
# skdata['S3ratio2'] = skdata['708.75']/ skdata['490']
bands_S3 = ['400', '412.5', '442.5', '490', '510', '560',
            '620', '665','673.75','681.25', '708.75','753.75', '778.75','865',
            '885']
# bands_S3ratio = ['S3ratio1','S3ratio2' ]
# bands_S3_plus_ratios = bands_S3 + bands_S3ratio
bands_try=[("S3bands", bands_S3)]
##[('S3ratio1',['S3ratio1']),('S3ratio2',['S3ratio2']),("S3bands", bands_S3),
#             ('S3bands&ratios', bands_S3_plus_ratios),("ratios_S3",bands_S3ratio)]
#

##Validation dataset for testing C2X
valdata = pd .read_csv('HL_C2A_total_test.txt', sep='\t', na_values=' ')
valdata = valdata[valdata['Chl_comp_1'] == 1]

# valdata['S3ratio1'] = valdata['665']/ valdata['490']
# valdata['S3ratio2'] = valdata['708.75']/ valdata['490']

#Subset cdom values < 0.3
# skdata = skdata[skdata['a_440_cdo'] > 0.3]
# valdata = valdata[valdata['a_440_cdo'] > 0.3]

# print(len(skdata), max(skdata['TSM']),min(skdata['TSM']),max(skdata['a_440_cdom']),min(skdata['a_440_cdom']),
#      max(skdata['TSM']),min(skdata['TSM']) )
# print(len(valdata), max(valdata['TSM']), min(valdata['TSM']), max(valdata['a_440_cdom']), min(valdata['a_440_cdom']),
#       max(valdata['TSM']), min(valdata['TSM']))
#

#
# ##S3
# datalines = pd.DataFrame(skdata.ix[:, 18:37])
# #print(datalines)
# datalines = datalines.transpose()
# # print(datalines.index, datalines.columns)
# ax1 = datalines.plot(x=datalines.index, y=datalines.columns, legend=False, grid=None)
# ax1.set_title('Simulation spectra C2A(X) OLCI', size=14)
# ax1.set(xlabel="Wavelength (nm)", ylabel="Rrs")
# ax1.tick_params(labelsize=10)
# ax1.patch.set_facecolor('white')
# plt.xticks(rotation='vertical')
# plt.subplots_adjust(bottom=0.25)
# # plt.show()
# plt.savefig('C2A_total_reflec_03.png')
# plt.close()

# ##SYKE dataset
# skdata['S2ratio1'] = skdata['S2665']/skdata['S2490']
# skdata['S2ratio2'] = skdata['S2705']/skdata['S2490']
# skdata['S3ratio1'] = skdata['S3665']/skdata['S3490']
# skdata['S3ratio2'] = skdata['S3708.75']/skdata['S3490']
#
# ###2.1 Other possibile inputs
# bands_S2 = ['S2443','S2490','S2560','S2665','S2705','S2740']
# bands_S3 = ['S3400','S3412.5','S3442.5','S3490','S3510','S3560',
#                   'S3620','S3665','S3673.75','S3681.25','S3708.75', 'S3753.75']
# bands_S2ratio = ['S2ratio1','S2ratio2']
# bands_S3ratio = ['S3ratio1','S3ratio2']
# bands_S2_plus_ratios = bands_S2 + bands_S2ratio
# bands_S3_plus_ratios = bands_S3 + bands_S3ratio
#
# #print(skdata.columns)
#
# bands_try=[('S2ratio1',['S2ratio1']),('S2ratio2',['S2ratio2']),("ratios_S2",bands_S2ratio),("S2bands",bands_S2),
#            ("S2bands&ratios", bands_S2_plus_ratios),
#            ('S3ratio1',['S3ratio1']),('S3ratio2',['S3ratio2']),("ratios_S3",bands_S3ratio),("S3bands",bands_S3),
#            ("S3bands&ratios", bands_S3_plus_ratios)]
#

# ##Plot ratio against a400
# # parameter_column= ['S2ratio1','S2ratio2','S3ratio1','S3ratio2']
# parameter_column = ['S3ratio1','S3ratio2']
# skdataSub = skdata[skdata['a_440_cdom'] > 0.3]
# valdataSub = valdata[valdata['a_440_cdom'] > 0.3]
# for parameter_col in parameter_column:
#     f, ax = plt.subplots(1, 1, sharey=True)
#     ax.patch.set_facecolor('white')
#     ax.xaxis.grid(color='lightgrey', linestyle='dotted')
#     ax.yaxis.grid(color='lightgrey', linestyle='dotted')
#     sns.regplot(x=parameter_col, y='a_440_cdom' , data=valdataSub, scatter_kws={"marker": "o", "color": "slategrey"},
#                 line_kws={"linewidth": .3, "color": "seagreen", "linestyle": "-"}, fit_reg=False, ax=ax)
#     ax.set(ylabel= 'a_440_cdom', xlabel= parameter_col)
#     # ax.set_title(parameter_col+ ' vs. a400 nm')
#     ax.set_title(parameter_col+ ' vs. a440 nm VAL')
#     ax.set(ylim=(0,30), xlim=(0, 8))##only for 670nm
#     # extra = plt.Rectangle((0, 0), 0.01, 0.01, fc="w", fill=False, edgecolor='none', linewidth=0.2)
#     # ax.legend([extra, extra, extra, extra, extra, extra, extra, extra, extra, extra],
#     #           (r'$slope=$' + a, r'$intercept=$' + b, r'$r2=$' + c,
#     #            r'$RMSE=$' + d, r'$bias=$' + g, r'$MAE=$' + l, r'$N=$' + k), loc='right')
#
#     f.set_tight_layout(False)
#     f.savefig(parameter_col+ '_polyfit_C2X_val_03.png')
#     # plt.show()
#     plt.close()
#

###For C2X dataset
# cdom_is = skdata['a400 (1/m)']
#

tabla_errores_totales =[]
for name_bands, bands in bands_try:
    if True:
        X_train = skdata[bands]
        X_train = X_train.as_matrix()
        cdom_array = skdata['TSM'].values
        y_train = cdom_array


        ##For C2X dataset
        X_test = valdata[bands]
        X_test = X_test.as_matrix()
        cdom_array_val = valdata['TSM'].values
        y_test = cdom_array_val

    else:
        X = skdata[bands]
        cdom_array = skdata['a400 (1/m)'].values
        X_train, X_test, y_train, y_test = train_test_split(X, cdom_array, test_size=0.25, random_state=42)


    ###3. Scale to [0, 1] range
    min_max_scaler = MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)



    X_test_scaled = min_max_scaler.transform(X_test)
    # X_test = X_test.as_matrix()
    # X_train = X_train.as_matrix()

    print( "Shapes train test:",X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(np.mean(y_train), np.mean(y_test))

    ###5. Remove the mean of Y for training only
    mean_y = np.mean(y_train)
    y_train_norm = y_train - mean_y
    y_test_norm = y_test - mean_y

    ###For C2X dataset
    X_test_scaled_df = pd.DataFrame(X_test_scaled,columns=bands)
    print(np.mean(y_train), np.mean(y_train_norm))

    ###5. Testing models
    ####models on a loop
    ## prepare models
    models = []
    verbose = 1
    n_jobs = 6
    cache_size = 4000
    kfold_hyperparams = KFold(n_splits= 10,random_state=12)
    models.append(('RLR', GridSearchCV(Ridge(),
                                      param_grid={ "alpha":np.array( [0, .0001, .001, .01, .1, 1. ])/X_train.shape[1]},
                                       cv= kfold_hyperparams,verbose= verbose,n_jobs=n_jobs)))
    models.append(('RFR', GridSearchCV(RandomForestRegressor(),
                                       param_grid={ "n_estimators":[ 10, 20, 50, 100,200  ]},  # "max_depth":[2,3,4]
                                       verbose= verbose, cv= kfold_hyperparams, n_jobs=n_jobs)))
    gamma_bar = X_train.shape[1] /2*np.logspace( -6 ,6,num=50)
    models.append(('KRR', GridSearchCV(KernelRidge(kernel="rbf"),
                                       param_grid={ "alpha":np.array([ .0001, .001, .01, .1, 1. ])/X_train.shape[0],
                                                   "gamma": gamma_bar},
                                       verbose= verbose,cv= kfold_hyperparams,n_jobs=n_jobs)))
    ##models.append(('DTR', DecisionTreeRegressor()))
    kernel = ConstantKernel(1.) * RBF(length_scale=np.repeat(1.0,X_train.shape[1]), length_scale_bounds=(1e-2, 1e2)) \
             + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4)
    models.append(('GPR', gp))

    gamma_bar = X_train.shape[1] / 2 * np.logspace(-6, 6, num=12)
    models.append(('SVR',
                   GridSearchCV(SVR(cache_size=cache_size),
                                param_grid={"C": np.array( [1, 10, 100,1000.]),
                                            "epsilon": np.array([ .001, .005, .01, .05, .1, .2 ]
                                            )/(np.max( y_train_norm)-np.min(y_train_norm)),
                                            "gamma": gamma_bar},
                                cv= kfold_hyperparams,verbose=verbose,
                                n_jobs=n_jobs)))

    if len(bands) == 1:
        polinomial_model = make_pipeline(PolynomialFeatures(degree=2 ),LinearRegression())
        models.append(( 'Polyfit',polinomial_model))

    # evaluate each model in turn using boxplots
    results_mae = []
    results_residuals = []
    names = []
    dat_save = []
    predictions = []

    for name, model in models:
        print("-------------------------------------------------")
        print( "Fitting model: %s with bands: %s"%(name ,name_bands))
        print("-------------------------------------------------")
        start = time.time()
        model.fit(X_train_scaled ,y_train_norm)
        fit_time = time.time()

        joblib.dump(model, name+"_"+name_bands+'_TSM.pkl')

        ###For C2X dataset
        # preds = model.predict(X_test)
        preds = model.predict(X_test_scaled)
        preds+=mean_y
        predict_time = time.time( ) -fit_time
        fit_time -=start
        residuals = (preds - y_test)

        ###basic metrics
        error_abs = get_rmse_abs(preds, y_test)
        error_rel = get_rmse_rel(preds, y_test)
        bias = get_bias(preds, y_test)
        bias_rem = y_test - bias
        res_abs = get_rmse_abs(bias_rem, preds)
        pearsonr, pvalue_pearsonr = scipy.stats.pearsonr(preds, y_test)
        r2_ = r2_score(y_test, preds)

        mae = mean_absolute_error(preds, y_test)

        # data for boxplots
        results_mae.append(np.abs(preds - y_test))
        results_residuals.append(preds - y_test)
        names.append(name)
        predictions.append(preds)

        if hasattr(model, "best_estimator_"):
            print(model.best_estimator_)
            best_params = model.best_params_
        elif hasattr(model, "kernel_"):
            print(model.kernel_)
            best_params = model.kernel_.get_params()
        else:
            print("No best params will be saved")
            best_params = {}

        dat_save.append({"rmse_abs": error_abs,
                         "rmse_rel": error_rel, "bias": bias, "res_abs": res_abs,
                         "mae": mae, "R2": r2_, "pearsonr": pearsonr,
                         "model": name, "best_params": best_params})



        msg = "%s %s: MAE:%f" % (name_bands, name, mae)
        print(msg)
        msg = "%s %s: fit time: %.3f (min) predict time: %.3f (min)" % (
        name_bands, name, fit_time / 60, predict_time / 60)
        print(msg)

        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(y_test, preds)

        a = str("{0:.5f}".format(slope))
        b = str("{0:.5f}".format(intercept))
        c = str("{0:.5f}".format(r_value))
        d = str("{0:.1E}".format(std_err))

        stats = pd.DataFrame(
            {'Slope': [a], 'Intercept': [b], 'r': c, 'error': [d]})

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(y_test, preds)
        # plt.ylabel("CDOM predicted")
        # plt.xlabel("CDOM measured")
        plt.ylabel("TSM predicted")
        plt.xlabel("TSM measured")
        ax.xaxis.grid(color='lightgrey', linestyle='dashed')
        ax.yaxis.grid(color='lightgrey', linestyle='dashed')
        ax.patch.set_facecolor('white')
        ax.set_title("Validation_" + name + "_" + name_bands)
        # ax.set_xlim(-5, 30)
        # ax.set_ylim(-5, 30)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="lightgrey")
        fig.text(0.2, 0.8, r'$y=$' + a + r'$x$' + b, ha='left', fontsize=10)
        fig.text(0.2, 0.76, r'$r=$' + c, ha='left', fontsize=10)
        fig.text(0.2, 0.72, r'$error=$' + d, ha='left', fontsize=10)

        # plt.show()
        fig.savefig("VALIDATION_" + name + "_" + name_bands + "_TSM.pdf")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(residuals, preds)
        ax.xaxis.grid(color='lightgrey', linestyle='dashed')
        ax.yaxis.grid(color='lightgrey', linestyle='dashed')
        ax.patch.set_facecolor('white')
        # ax.set_xlim(-10, 20)
        # ax.set_ylim(-10, 20)
        plt.ylabel("Residuals")
        plt.xlabel("TSM predicted")
        # ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        ax.set_title("Residuals_" + name + "_" + name_bands)
        # plt.show()
        fig.savefig("RESIDUALS_" +  name + "_" + name_bands + "_TSM.pdf")

        if len(bands) > 2:
            # print("      Computing partial plots")
            # XPLOTS, PPLOTS = partial_plots(model, X_test_df)
            # change depending of number of subplots and name of parameters
            ##OLCI
            # fig, ax = plt.subplots(4, 3)
            ##OLCI 21 bands
            fig, ax = plt.subplots(5, 4)
            ##MSI
            # fig, ax = plt.subplots(3, 2)
            ax = ax.flatten()

            '''
            for axe, col in zip(ax, XPLOTS.columns):
                axe.plot(XPLOTS[col], PPLOTS[col])
                axe.scatter(X_test_df[col], y_test, c="grey", marker="+", s=10)
                axe.tick_params(axis='both', which='major', labelsize=10)
                axe.xaxis.grid(color='lightgrey', linestyle='dotted')
                axe.yaxis.grid(color='lightgrey', linestyle='dotted')
                axe.patch.set_facecolor('white')
                plt.xticks(rotation='vertical')
                plt.subplots_adjust(bottom=0.25)

            ax[0].set_title('400 nm', fontsize=10)
            ax[1].set_title('412 nm', fontsize=10)
            ax[2].set_title('443 nm', fontsize=10)
            ax[3].set_title('490 nm', fontsize=10)
            ax[4].set_title('510 nm', fontsize=10)
            ax[5].set_title('560 nm', fontsize=10)
            ax[6].set_title('620 nm', fontsize=10)
            ax[7].set_title('665 nm', fontsize=10)
            ax[8].set_title('753 nm', fontsize=10)
            ax[9].set_title('778 nm', fontsize=10)
            ax[10].set_title('865 nm', fontsize=10)



            # ax[8].set_title('673 nm', fontsize=10)
            # ax[9].set_title('681 nm', fontsize=10)
            # ax[10].set_title('708 nm', fontsize=10)
            # ax[11].set_title('753 nm', fontsize=10)



            fig.subplots_adjust(hspace=0.4)
            fig.suptitle(name_bands + "_" + name, fontsize=10)
            '''

            # plt.savefig("partial_plots_"+name_bands+"_"+name+".pdf")


            fig = plt.figure()
            print("      Computing permutation test")
            sal = permutation_test(model, X_test_scaled_df, y_test - mean_y)
            sns.barplot(x="variable", y="value", data=pd.melt(sal))
            plt.xticks(rotation='vertical')
            plt.subplots_adjust(bottom=0.25)
            plt.title(name_bands + "_" + name)
            plt.savefig("permutation_" + name_bands + "_" + name + "_TSM.pdf")

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
        fig.savefig("regression_" + name_bands + "_TSM.pdf")


    dat_save = pd.DataFrame(dat_save)

    dat_save.to_csv(name_bands + "_TSM.csv", index=False)

    predictions = pd.DataFrame(predictions).T
    # predictions.to_csv(name_bands + "_preds.csv", index=False)


    dat_save["name_bands"] = name_bands
    predictions["name_bands"] = name_bands
    tabla_errores_totales.append(dat_save)


    # save boxplot
    fig = plt.figure()
    # fig.suptitle('Algorithm Comparison MAE')
    ax = fig.add_subplot(111)
    plt.boxplot(results_mae)
    ax.set_ylim(-5, 5)
    ax.set_xticklabels(names)
    ax.set_title("MAE_" + name_bands)
    ax.xaxis.grid(color='lightgrey', linestyle='dashed')
    ax.yaxis.grid(color='lightgrey', linestyle='dashed')
    ax.patch.set_facecolor('white')
    fig.savefig("MAE_" + name_bands + "_TSM.pdf")

    fig = plt.figure()
    # fig.suptitle('Algorithm Comparison residuals')
    ax = fig.add_subplot(111)
    plt.boxplot(results_residuals)
    ax.set_ylim(-10, 10)
    ax.set_xticklabels(names)
    ax.set_title("ERRORS_" + name_bands)
    ax.xaxis.grid(color='lightgrey', linestyle='dashed')
    ax.yaxis.grid(color='lightgrey', linestyle='dashed')
    ax.patch.set_facecolor('white')
    fig.savefig("ERRORS_" + name_bands + "_TSM.pdf")


tabla_errores_totales = pd.concat(tabla_errores_totales, ignore_index=True)
tabla_errores_totales = pd.DataFrame(tabla_errores_totales)
tabla_errores_totales.to_csv("tabla_errores_totales_C2AX_TSM.csv", index=False)




