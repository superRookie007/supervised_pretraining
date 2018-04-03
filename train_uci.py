from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import utils
from keras import backend as K
import sklearn.metrics as metrics



DATA_NAME = "waveform"
UNSHUFFLED_LR = 0.01
PRETRAIN_LR = 0.01
LAST_LR = 0.01
DECAY = 0.0
EPOCHS = 1
WIDTH = 1024
DEPTH = 5
ACTIVATION = 'relu'
RUNS = 1

cwd = os.path.dirname(os.path.realpath(__file__))
TRAIN_PATH = "{}/uci_data/{}/train.csv".format(cwd, DATA_NAME)
TEST_PATH = "{}/uci_data/{}/test.csv".format(cwd, DATA_NAME)
PRE_TRAIN_PATH = "{}/uci_data/{}/pre_train.csv".format(cwd, DATA_NAME)
PRE_VAL_PATH = "{}/uci_data/{}/pre_val.csv".format(cwd, DATA_NAME)
OUTPUT_PATH = "{}/uci_data/{}.txt".format(cwd, DATA_NAME+"_results")


# Load data to numpy arrays
x_train, y_train = utils.load_csv(TRAIN_PATH)
x_test, y_test = utils.load_csv(TEST_PATH)
x_pre_train, y_pre_train = utils.load_csv(PRE_TRAIN_PATH)
x_pre_val, y_pre_val = utils.load_csv(PRE_VAL_PATH)

# get number of classes
num_classes = y_train.shape[1]



# get number of instances and input dimension for both training and test sets
total_train_instances, train_input_dim = x_train.shape
total_test_instances, test_input_dim = x_test.shape
# training and test data should have the same dimensions
assert train_input_dim == test_input_dim
assert y_train.shape[1] == y_test.shape[1]
num_classes = y_train.shape[1]

# normalize
scaler = MinMaxScaler(copy=False)
scaler.fit_transform(x_pre_train)
scaler.transform(x_test)
scaler.transform(x_train)
scaler.transform(x_pre_val)


###################################################################################
weights_data_folder = "{}/tmp/weights_uci/{}".format(cwd, DATA_NAME)
try:
    os.makedirs(weights_data_folder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

for run in range(RUNS):

    run_folder = "{}/run{}".format(weights_data_folder, run+1)
    try:
        os.makedirs(run_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    INIT = keras.initializers.he_normal()

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1, mode='auto') #50
    terminateOnNaN = keras.callbacks.TerminateOnNaN()

    with open(OUTPUT_PATH, 'a') as output_file:
        output_file.write("##### Run %d #####\n" % (run+1))

#################################################################################################################################3

    print("##### Base #####")
    model = Sequential()
    model.add(Dense(
        units=WIDTH,
        activation=ACTIVATION,
        kernel_initializer=INIT,
        input_shape=(train_input_dim,)))
    for i in range(1, DEPTH-1):
        model.add(Dense(units=WIDTH, activation=ACTIVATION, kernel_initializer=INIT))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=INIT))
    sgd = keras.optimizers.SGD(lr=UNSHUFFLED_LR, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x_train,
        y_train,
        batch_size=total_train_instances,
        epochs=EPOCHS,
        verbose=1,
        # callbacks=[terminateOnNaN],
        validation_data=(x_test, y_test),
        shuffle=False,
        initial_epoch=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    prediction_scores = model.predict(x_test)
    roc_auc_score = metrics.roc_auc_score(y_test, prediction_scores, average='macro')

    test_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(prediction_scores, axis=1)

    f_score = metrics.f1_score(test_labels, predicted_labels, average='macro')
    cohen_kappa_score = metrics.cohen_kappa_score(test_labels, predicted_labels)

    print("\nStopped at Epoch %05d: {test_loss: %f; test_acc: %f; auc_roc: %f; f_score: %f; cohen_kappa_score: %f}" % (EPOCHS, test_loss, test_acc, roc_auc_score, f_score, cohen_kappa_score))
    with open(OUTPUT_PATH, 'a') as output_file:
        output_file.write("----- Base -----\n")
        output_file.write("Base: stopped_at_Epoch %d; train_loss %f; train_acc %f, test_loss: %f; test_acc: %f; auc_roc: %f; f_score: %f; cohen_kappa_score: %f\n\n" % (EPOCHS, history.history['loss'][-1], history.history['acc'][-1], test_loss, test_acc, roc_auc_score, f_score, cohen_kappa_score))
    del model
    K.clear_session()


    print("##### Pre-training #####")
    model = Sequential()
    model.add(Dense(
        units=WIDTH,
        activation=ACTIVATION,
        kernel_initializer=INIT,
        input_shape=(train_input_dim,)))
    for i in range(1, DEPTH-1):
        model.add(Dense(units=WIDTH, activation=ACTIVATION, kernel_initializer=INIT))
    model.add(Dense(2, activation='softmax', kernel_initializer=INIT))
    sgd = keras.optimizers.SGD(lr=PRETRAIN_LR, momentum=0.0, decay=DECAY, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    path_to_weights = "{}/pretrain_initial_weights.hdf5".format(run_folder)
    check_point = keras.callbacks.ModelCheckpoint(path_to_weights, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='max', period=1)

    # ### Pre-training to get initial weights
    model.fit(
        x_pre_train,
        y_pre_train,
        batch_size=256,
        epochs=100000,
        verbose=1,
        callbacks=[check_point, early_stopping, terminateOnNaN],
        validation_data=(x_pre_val, y_pre_val),
        shuffle=True,
        initial_epoch=0)
    val_loss, val_acc = model.evaluate(x_pre_val, y_pre_val, verbose=1)
    with open(OUTPUT_PATH, 'a') as output_file:
        output_file.write("----- Pre-training -----\n")
    del model
    K.clear_session()


    # Start training
    print("----- Training -----")
    path_to_weights = "{}/pretrain_initial_weights.hdf5".format(run_folder)

    model = Sequential()
    model.add(Dense(
        units=WIDTH,
        activation=ACTIVATION,
        kernel_initializer=INIT,
        input_shape=(train_input_dim,)))
    for i in range(1, DEPTH-1):
        model.add(Dense(units=WIDTH, activation=ACTIVATION, kernel_initializer=INIT))
    model.add(Dense(2, activation='softmax', kernel_initializer=INIT))
    model.load_weights(path_to_weights, by_name=False)
    # remove last n layers
    model.layers = model.layers[0:DEPTH-1]
    # add new layer
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=INIT))
    assert len(model.layers) == DEPTH


    sgd = keras.optimizers.SGD(lr=LAST_LR, momentum=0.0, decay=DECAY, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        batch_size=total_train_instances,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[terminateOnNaN],
        validation_data=(x_test, y_test),
        shuffle=False,
        initial_epoch=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    prediction_scores = model.predict(x_test)
    roc_auc_score = metrics.roc_auc_score(y_test, prediction_scores, average='macro')

    test_labels = np.argmax(y_test, axis=1)
    predicted_labels = np.argmax(prediction_scores, axis=1)

    f_score = metrics.f1_score(test_labels, predicted_labels, average='macro')
    cohen_kappa_score = metrics.cohen_kappa_score(test_labels, predicted_labels)

    print("\nStopped at Epoch %d: {test_loss: %f; test_acc: %f; auc_roc: %f; f_score: %f; cohen_kappa_score: %f}" % (EPOCHS, test_loss, test_acc, roc_auc_score, f_score, cohen_kappa_score))
    with open(OUTPUT_PATH, 'a') as output_file:
        output_file.write("----- Training -----\n")
        output_file.write("Pretrained: stopped_at_Epoch %d; train_loss %f; train_acc %f, test_loss: %f; test_acc: %f; auc_roc: %f; f_score: %f; cohen_kappa_score: %f\n\n" % (EPOCHS, history.history['loss'][-1], history.history['acc'][-1], test_loss, test_acc, roc_auc_score, f_score, cohen_kappa_score))
    del model
    K.clear_session()
