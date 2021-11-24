## Utils functions

from sklearn.decomposition import PCA
import numpy as np

# Function to add noise to improve the visualization and interpretation of the data.
def jitter(data, sigma=0.3):
    random_sign = (-1.)**np.random.randint(1, high=3, size=data.shape)
    return data + np.random.normal(0, sigma, size=data.shape) * random_sign

# Function to separate training data from validation data. Random.
def single_stratified_split(X, Y, test_size=.2, random_state=1234):
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    split_ix = splitter.split(X, Y)
    for train_ix, test_ix in split_ix:
        X_train = X.loc[train_ix].reset_index(drop=True)
        Y_train = Y.loc[train_ix].reset_index(drop=True)
        X_test = X.loc[test_ix].reset_index(drop=True)
        Y_test = Y.loc[test_ix].reset_index(drop=True)
    return X_train, Y_train, X_test, Y_test


# Function to adjust the image to the number.
def number_adjust(img):
    # Width
    col = img.sum(axis=0)
    indc = np.argwhere(col > 0)
    # Height
    row = img.sum(axis=1)
    indr = np.argwhere(row > 0)
    img_rec = img[int(indr[0]):int(indr[-1]), int(indc[0]):int(indc[-1])]
    return img_rec


# PCA
def our_PCA(data, n_components=.7):
    n_components = n_components
    if n_components <= data.shape[1]:
        pca = PCA(n_components = n_components)
        pca.fit(data)
        X_proy = pca.transform(data)
        print(X_proy.shape)
    else:
        print("ERROR: the number of principal components has to be less or equal than data dimension !")
    return X_proy, pca

