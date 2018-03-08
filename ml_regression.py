__author__ = 'Gonzalo Mateo-García, Ana Ruescas'

import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from ml import models
from ml import data_load

def train(path_to_save_models=data_load.PATH_TO_MODELS_SYKE,
               bands_try=list(data_load.bands_try_SYKE.keys()),
               name_models_try=models.ALL_NAME_MODELS,n_jobs=-1):

    ###1. Read dataset
    skdata_X_train, _, skdata_y_train, _ =  data_load.load_SYKE()
    mean_y = np.mean(skdata_y_train)
    skdata_y_train_norm = skdata_y_train - mean_y


    for name_bands in bands_try:
        bands = data_load.bands_try_SYKE[name_bands]
        print("Trying bands %s"%name_bands)

        X_train = skdata_X_train[bands]

        ###3. Scale to [0, 1] range
        min_max_scaler = MinMaxScaler()
        X_train_scaled = min_max_scaler.fit_transform(X_train)

        print( "Shapes train:",X_train.shape)

        modelos = models.load_models(X_train,skdata_y_train_norm,n_jobs=n_jobs)

        models_try = {}
        for name,model in modelos.items():
            if name in name_models_try:
                models_try[name] = model

        assert bool(models_try), "There are no models to fit"

        fitting_times = models.fit_models(models_try, X_train_scaled, skdata_y_train_norm)

        for name, model in models_try.items():
            if hasattr(model, "best_estimator_"):
                model = model.best_estimator_
            model_save = make_pipeline(min_max_scaler, model)
            joblib.dump(model_save, os.path.join(path_to_save_models, name + "_" + name_bands + '.pkl'))


if __name__ == "__main__":
    print("njobs %d"%n_jobs)
    train(n_jobs=n_jobs)