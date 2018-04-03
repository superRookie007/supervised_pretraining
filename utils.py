from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.model_selection import StratifiedShuffleSplit


# ###  loading the data
def load_csv(path):
    """
    Read data in csv format.
    Note: this function should only be used for dataset that can fit into memory.

    Args:
        path: a string that specifies the path to the csv file

    Return:
        data as numpy arrays
    """

    # load raw data into memory
    data = pd.read_csv(path, header=None)

    # find the target index from the target location index
    target_index = data.columns[-1]

    labels = pd.get_dummies(data[target_index])
    data.drop(target_index, axis=1, inplace=True)

    return data.as_matrix(), labels.as_matrix()

def permute_columns(x):
    "Shuffle the feature values"
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]


def in_original(array, original):
    "Check if array is in a different array 'original'"
    for i in range(original.shape[0]):
        if np.array_equal(array, original[i]):
            return True
    return False


def permute_columns_plus(x):
    "Shuffle the feature values and make sure that the shuffled data does not contain any data points in the original data"
    count = 0
    shuffled = np.empty(shape=x.shape)
    while count != x.shape[0]:
        temp = permute_columns(x)
        for i in range(x.shape[0]):
            if not in_original(temp[i], x):
                shuffled[i] = temp[i]
                count += 1
                if count == x.shape[0]:
                    break
    return shuffled

def get_pretrain_data(path, seed=None):
    """
    Generate the data for supervised pretraining.
    The pretraining data has binary label. The original data is labelled as True,
    while the shuffled data is labelled as False.
    This function makes sure that the shuffled data does not contain any data points in the original data.

    Args:
        path: string. This is the path to the unlabelled data in csv format.
        seed: integer. Set the seed for shuffling. Default to be None.

    Return:
        pretrain

    """
    np.random.seed(seed)
    count = 0
    unlabelled = pd.read_csv(path, header=None).as_matrix()

    # create pretrain data
    false = np.empty(shape=unlabelled.shape)
    while count != unlabelled.shape[0]:
        shuffled = permute_columns(unlabelled)
        for i in range(unlabelled.shape[0]):
            if not in_original(shuffled[i], unlabelled):
                false[i] = shuffled[i]
                count += 1
                if count == unlabelled.shape[0]:
                    break
    x_pretrain = np.concatenate((unlabelled, false), axis=0)

    # create pretrain labels
    y_pretrain_real = np.ones((unlabelled.shape[0], 1))
    y_pretrain_shuffled = np.zeros((false.shape[0], 1))
    y_pretrain = np.concatenate((y_pretrain_real, y_pretrain_shuffled), axis=0)

    assert x_pretrain.shape[0] == y_pretrain.shape[0]
    pretrain = np.concatenate((x_pretrain, y_pretrain), axis=1)

    return pretrain


def f_score(y_true, y_pred):
    "Compute the F-score using gound truth and predicted labels."
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # calculate precision and recall
    precision = c1 / c2
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def shuffle_pixels(images):
    "Shuffle the pixels of each image"
    flat = images.reshape((images.shape[0], -1))
    np.apply_along_axis(np.random.shuffle, 1, flat)
    return

def shuffle_pixels_across_images(images):
    "Shuffle the pixels of each image"
    flat = images.reshape((images.shape[0], -1))
    shuffled = permute_columns(flat)
    return shuffled.reshape(images.shape)

def stratified_subset(data, target, size, seed):
    """
    Create a subset of data of size "size" using stratification based on target.
    Args:
        data: array-like, shape (n_samples, n_features)
        target: array-like, shape (n_samples,)
        size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the subset. If int, represents the absolute number of samples.
        seed: int
    Return:
        a tuple of subset of data and subset of target.
    """
    split = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed)
    for train_index, _ in split.split(data, target):
        subdata = data[train_index]
        sublabel = target[train_index]
    return (subdata, sublabel)
