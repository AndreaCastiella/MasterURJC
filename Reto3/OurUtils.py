## Utils functions

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, MinMaxScaler
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


def our_kernelPCA(data, n_components=5):
    n_components = n_components
    kernel = "rbf" # options are: "linear", "poly", "rbf", "sigmoid"
    kernel_parameter = 1
    rbf_pca = KernelPCA(n_components = n_components,
                        kernel=kernel, gamma=kernel_parameter, fit_inverse_transform=True)
    X_proy = rbf_pca.fit_transform(data)
    return X_proy

def our_scale_transform_features(X, Y, scaler=None, use_pca=False, transform=None, transform_type="transformer"):
    
    # scale features
    if scaler == None:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    # pca
    pca = None
    if use_pca:
        X, pca = our_PCA(X)

    # apply some preproccesing
    if transform is not None:
        X = transform.fit_transform(X)
    else:
        if transform_type == "poly":
            transform = PolynomialFeatures(2)
            X = transform.fit_transform(X)

        if transform_type == "transformer":
            transform = FunctionTransformer(np.log1p, validate=True)
            X = transform.fit_transform(X)

    # get Y values
    Y = Y.values.ravel()
    return (X, Y, scaler, pca, transform)
