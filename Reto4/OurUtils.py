from sklearn.model_selection import StratifiedShuffleSplit
import cv2

def single_stratified_split(X, Y, test_size=.2, random_state=1234):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    split_ix = splitter.split(X,Y)
    for train_ix, test_ix in split_ix:
        X_train = X[train_ix]
        Y_train = Y[train_ix]
        X_test = X[test_ix]
        Y_test = Y[test_ix]
    return X_train, Y_train, X_test, Y_test