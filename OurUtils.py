## Utils functions

import pandas as pd
import numpy as np

# Función para añadir ruido para mejorar la visualización e interpretación de los datos.
def jitter(data, sigma=0.3):
    random_sign = (-1.)**np.random.randint(1, high=3, size=data.shape)
    return data + np.random.normal(0, sigma, size=data.shape)*random_sign

# Función para separar los datos de entrenamiento de los datos de validación. Random.
def single_stratified_split(X, Y, test_size=.2, random_state=1234):
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    split_ix = splitter.split(X,Y)
    for train_ix, test_ix in split_ix:
        X_train = X.loc[train_ix].reset_index(drop=True)
        Y_train = Y.loc[train_ix].reset_index(drop=True)
        X_test  = X.loc[test_ix].reset_index(drop=True)
        Y_test  = Y.loc[test_ix].reset_index(drop=True)
    return X_train, Y_train, X_test, Y_test

# Función para concatenar características con las etiquetas.
def join_features_labels(X0, X1):
    Y0 = pd.DataFrame(np.zeros(X0.shape[0]),columns=['label'])
    XY0 = pd.concat([X0,Y0],axis=1)
    Y1 = pd.DataFrame(np.ones(X1.shape[0]),columns=['label'])
    XY1 = pd.concat([X1,Y1],axis=1)
    return pd.concat([XY0,XY1],axis=0,ignore_index=True)
