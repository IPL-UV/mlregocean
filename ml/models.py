__author__ = 'Gonzalo Mateo-García, Ana Ruescas'

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import  GridSearchCV ,KFold
from sklearn.linear_model import Ridge ,LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import time

ALL_NAME_MODELS = ['RLR','RFR','KRR','SVR','GPR','Polyfit']
def load_models(X_train, y_train, n_jobs=6):
    verbose = 1
    cache_size = 4000
    models = []
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
                                            )/(np.max( y_train)-np.min(y_train)),
                                            "gamma": gamma_bar},
                                cv= kfold_hyperparams,verbose=verbose,
                                n_jobs=n_jobs)))

    if X_train.shape[1] == 1:
        polinomial_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        models.append(('Polyfit', polinomial_model))

    return dict(models)

def fit_models(models, X_train, y_train):

    fitting_time = {}
    for name, model in models.items():
        print("-------------------------------------------------")
        print("Fitting model: %s"%name)
        print("-------------------------------------------------")
        start = time.time()
        model.fit(X_train, y_train)
        fitting_time[name] = time.time()

    return fitting_time
