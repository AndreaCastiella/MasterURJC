## Utils functions

import pandas as pd
import numpy as np

# Function to add noise to improve the visualization and interpretation of the data.
def jitter(data, sigma=0.3):
    random_sign = (-1.)**np.random.randint(1, high=3, size=data.shape)
    return data + np.random.normal(0, sigma, size=data.shape) * random_sign

# Function to separate training data from validation data. Random.
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

# Function to concatenate features with tags.
def join_features_labels(X0, X1, X2, X3):
    Y0 = pd.DataFrame(np.zeros(X0.shape[0]), columns=['label'])
    XY0 = pd.concat([X0, Y0], axis=1)
    Y1 = pd.DataFrame(np.ones(X1.shape[0]), columns=['label'])
    XY1 = pd.concat([X1, Y1], axis=1)
    Y2 = pd.DataFrame(np.ones(X2.shape[0]), columns=['label'])
    XY2 = pd.concat([X2, Y2], axis=1)
    Y3 = pd.DataFrame(np.ones(X3.shape[0]), columns=['label'])
    XY3 = pd.concat([X3, Y3], axis=1)

    return pd.concat([XY0, XY1, XY2, XY3], axis=0, ignore_index=True)

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


# Feature extraction
def feat_extraction(data, perc=0.3, perc2=0.45, alfa=0.5, adjust=True):
    num_feat = 9
    features = np.zeros([data.shape[0], num_feat])
    data = data.values.reshape([data.shape[0],28,28]) # Each row is an image, reshape a 28x28.

    for i in range(data.shape[0]): # For each image.
        img = data[i,:,:]
        if adjust:
            img = our.number_adjust(img) # Returns an image with the size adjusted to the number.
        # Característica 1
        img_left = img[:, :int(img.shape[1]*(perc))]
        feat_1 = np.sum(img_left > 0.5)/(img_left.shape[0]*img_left.shape[1]) # Percentage of pixels of the left % of the image > 128 (0.5).
        features[i, 0] = feat_1
        # Característica 2
        img_inf = img[int(img.shape[0]*(1-perc)):, :]
        # feat_2 = np.sum(img_inf) # Sum of the pixels of the lower % of the image.
        feat_2 = np.sum(img_inf > 0.5)/(img_inf.shape[0]*img_inf.shape[1]) # Percentage of pixels of the lower % of the image > 128 (0.5).
        features[i, 1] = feat_2
        # Característica 3
        img_right = img[:, int(img.shape[1]*(1-perc)):]
        # feat_3 = np.sum(img_der) # Sum of the pixels of the right % of the image.
        feat_3 = np.sum(img_right > 0.5)/(img_right.shape[0]*img_right.shape[1]) # Percentage of pixels of the right % of the image > 128 (0.5).
        features[i, 2] = feat_3
        # Característica 4
        img_sup = img[:int(img.shape[0]*(perc)), :]
        # feat_4 = np.sum(img_sup) # Sum of the pixels in the upper % of the image
        feat_4 = np.sum(img_sup > 0.5)/(img_sup.shape[0]*img_sup.shape[1]) # Percentage of pixels of the upper % of the image > 128 (0.5).
        features[i, 3] = feat_4
        # Característica 5
        img_cuad = img[int(img.shape[0]*(1-perc)):,int(img.shape[1]*(perc)):]
        # feat_5 = np.sum(img_cuad) # Sum of the lower right quadrant
        feat_5 = np.sum(img_cuad > 0.5)/(img_cuad.shape[0]*img_cuad.shape[1]) # Percentage of pixels of the lower right quadrant % of the image > 128 (0.5).
        features[i, 4] = feat_5
        # Característica 6
        feat_6 = np.amax(np.sum(img, axis=0)) # Maximum value of the sum of the columns.
        features[i, 5] = feat_6
        # Característica 7
        img_inf = img[int(img.shape[0]*(1-perc)):, :] # Lower %.
        sum_cols = img_inf.sum(axis=0)
        indc = np.argwhere(sum_cols > alfa * sum_cols.max())
        feat_7 = indc[-1] - indc[0] # Theta-dependent width of the lower % of the image.
        features[i, 6] = feat_7
        # Característica 8
        img_inf = img[int(img.shape[0]*(1-perc)):, :] # Lower %.
        sum_rows = img_inf.sum(axis=1)
        indr = np.argwhere(sum_rows > alfa * sum_rows.max())
        feat_8 = indr[-1] - indr[0] # High dependent on theta of the lower % of the image.
        features[i, 7] = feat_8
        # Caracteristica 9
        img_upper = img[:int(img.shape[0] * (perc2)), :]
        sum_rows = img_upper.sum(axis = 1)
        idx_row = sum_rows.argmax()
        img_row = img_upper[idx_row, :]
        args = np.argwhere( img_row > 0.6)
        distance_upper = args[-1,0] - args[0,0]
        if distance_upper == 0:
            distance_upper = 0.01
        img_lower = img[int(img.shape[0] * (1-perc2)):, :]
        sum_rows = img_lower.sum(axis = 1)
        idx_row = sum_rows.argmax()
        img_row = img_lower[idx_row, :]
        args = np.argwhere( img_row > 0.6)
        distance_lower = args[-1,0] - args[0,0]
        features[i, 8] = float(distance_lower)/float(distance_upper) # Max upper distance / max lower distance
    col_names = ['feat_1','feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9']
    return pd.DataFrame(features,columns = col_names)

