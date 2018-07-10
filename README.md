# Machine Learning Regression for Ocean Parameters retrieval

Regression approaches for ocean parameter retrievals.

This repository contains the code used in the paper: *"Machine learning regression approaches for colored dissolved organic matter (CDOM) retrieval with Sentinel 2 and Sentinel 3 data"*.

The repository contains:
* The script `ml_regression.py` train and save the proposed models with the different combination of bands.
* The notebook `results.ipynb` evaluates the different trained models.  
* The notebook `cdom_predict_image.ipynb` loads a Sentinel-3 image and apply a trained model to the data.
* Some helper functions used by the aforementioned scripts on the folder `ml`.


