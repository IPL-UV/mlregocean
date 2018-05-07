from ml import cdom_processing
import numpy as np
import os
from sklearn.externals import joblib
from ml import data_load, models
import time
import json

N_TIMES = 10

if __name__== "__main__":
    s3image_path = "/media/disk/databases/C2X/S3Images/S3A_OL_1_EFR____20160524T090953_20160524T091153_20170930T000736_0119_004_264______MR1_R_NT_002.SEN3_C2RCC.data"
    img = cdom_processing.load_S3_image_C2RCC(image_path=s3image_path)
    mascara = cdom_processing.load_mask_C2RCC(s3image_path)
    _, _, skdata_y_train, _ = data_load.load_C2X()
    skdata_y_train = skdata_y_train["a_440_cdom"].values

    mean_y_train = np.mean(skdata_y_train)
    max_cdom = np.max(skdata_y_train)

    times_all = {}
    for bands_name in data_load.bands_try_C2X:
        image_predict = cdom_processing.image_to_predict_S3_C2RCC(img, data_load.bands_try_C2X[bands_name])
        for name_model in models.ALL_NAME_MODELS:
            file_model = os.path.join(data_load.PATH_TO_MODELS_C2X,
                                      "CDOM_" + name_model + '_' + bands_name + '.pkl')
            if not os.path.exists(file_model):
                print("File %s does not exist" % file_model)
                continue

            print("Timing model %s %s"%(name_model,bands_name))
            regressor_sklearn = joblib.load(file_model)

            times_model = []
            for i in range(N_TIMES):
                print("\t step %d/%d"%(i,N_TIMES))
                start = time.time()
                predictions = cdom_processing.predict_image(image_predict,
                                                            mascara,
                                                            regressor_sklearn,
                                                            y_range=(0, max_cdom),
                                                            y_mean=mean_y_train, step=300)
                times_model.append(time.time()-start)

            with open("times_"+name_model + '_' + bands_name+".json","w") as f:
                json.dump(times_model,f)

            times_all[name_model + '_' + bands_name] = times_model

    with open("times_C2X_all.json", "w") as f:
        json.dump(times_all, f)













