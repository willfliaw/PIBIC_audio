# utils/models.py

import logging
import os
import random

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Dense, Dropout, Flatten, MaxPooling1D)
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tqdm import tqdm

from utils.config import (BATCH_SIZE, EPOCHS, MODELS_PATH, N_SPLITS,
                          TRANSLATIONS)
from utils.feature_selection import dump_to_file

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))

def dnn_builder(feature_space, n_outputs):
  model = Sequential()

  model.add(Dense(units = 256,
                  input_shape = (len(feature_space.columns),)))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dropout(rate = 0.2))

  model.add(Dense(units = 128,
                  kernel_regularizer = regularizers.l2(0.001),
                  kernel_constraint = MaxNorm(max_value = 4)))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dropout(rate = 0.5))

  model.add(Dense(units = 64,
                  kernel_regularizer = regularizers.l2(0.001),
                  kernel_constraint = MaxNorm(max_value = 4)))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dropout(rate = 0.5))

  model.add(Dense(units = 32,
                  kernel_regularizer = regularizers.l2(0.001),
                  kernel_constraint = MaxNorm(max_value = 4)))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(Dropout(rate = 0.5))

  model.add(Dense(units = n_outputs, activation = "softmax"))

  model.compile(optimizer = Adam(learning_rate = 0.0001),
                loss = "categorical_crossentropy",
                metrics = ["accuracy"])

  return model

# def cnn1d_builder(feature_space, n_outputs):
#   model = Sequential()

#   model.add(Conv1D(128, kernel_size = (8),padding = "same", input_shape = (len(feature_space.columns), 1)))
#   model.add(BatchNormalization())
#   model.add(Activation("relu"))
#   model.add(MaxPooling1D(pool_size = (5)))
#   model.add(Dropout(0.5))

#   model.add(Conv1D(64, kernel_size = (8), padding = "same"))
#   model.add(BatchNormalization())
#   model.add(Activation("relu"))
#   model.add(MaxPooling1D(pool_size = (5)))
#   model.add(Dropout(0.5))

#   model.add(Flatten())

#   model.add(Dense(n_outputs, activation = "softmax"))

#   model.compile(optimizer = Adam(),
#                 loss = "categorical_crossentropy",
#                 metrics = ["accuracy"])

#   return model

def train_model(mod, feature_space, labels, save_dir, batch_size = BATCH_SIZE, epochs = EPOCHS):
  X_train, X_val, y_train, y_val = train_test_split(feature_space, labels, stratify = labels, test_size = 0.2, random_state = 0)
  X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.25, random_state = 0)

  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  X_val = scaler.transform(X_val)

  seed_value = 0
  os.environ["PYTHONHASHSEED"] = str(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value)
  tf.random.set_seed(seed_value)
  session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
  sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph(), config = session_conf)
  K.set_session(sess)

  if mod == "dnn":
    model = dnn_builder(feature_space, len(np.unique(np.argmax(labels, axis = 1))))
  # elif mod == "cnn1d":
  #   model = cnn1d_builder(feature_space, len(np.unique(labels)))
  #   X_train = np.expand_dims(X_train, axis = 2)
  #   X_test = np.expand_dims(X_test, axis = 2)
  #   X_val = np.expand_dims(X_val, axis = 2)
  # elif mod = "cnn2d":
  #     model = cnn2d_builder()

  ec = EarlyStopping(monitor = "val_loss", patience = 200, restore_best_weights = True)

  y_integers = np.argmax(y_train, axis = 1)
  class_weights = class_weight.compute_class_weight("balanced", classes = np.unique(y_integers), y = y_integers)
  d_class_weights = dict(enumerate(class_weights))

  history = model.fit(X_train, y_train,
                      validation_data = (X_test, y_test),
                      batch_size = batch_size,
                      epochs = epochs,
                      callbacks = [ec],
                      class_weight = d_class_weights,
                      verbose = 0)

  save_model(model, os.path.join(save_dir, mod))
  dump_to_file(os.path.join(save_dir, mod), "scaler", scaler)

  return history, model, X_val, y_val

def train_model_cv(mod, feature_space, labels, save_dir, n_splits = N_SPLITS, batch_size = BATCH_SIZE, epochs = EPOCHS):
  histories = []
  acc_per_fold = []
  loss_per_fold = []

  X_train, X_val, y_train, y_val = train_test_split(feature_space, labels, stratify = labels, test_size = 0.2, random_state = 0)

  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)

  kfold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 0)

  seed_value = 0
  os.environ["PYTHONHASHSEED"] = str(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value)
  tf.random.set_seed(seed_value)
  session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1)
  sess = tf.compat.v1.Session(graph = tf.compat.v1.get_default_graph(), config = session_conf)
  K.set_session(sess)

  if mod == "dnn":
    model = dnn_builder(feature_space, len(np.unique(np.argmax(labels, axis = 1))))
  # elif mod == "cnn1d":
  #   model = cnn1d_builder(feature_space, len(np.unique(labels)))
  #   X_train = np.expand_dims(X_train, axis = 2)
  #   X_val = np.expand_dims(X_val, axis = 2)
  # elif mod = "cnn2d":
  #     model = cnn2d_builder()

  for train, test in tqdm(kfold.split(X_train, np.argmax(y_train, axis = 1)), ncols = 50):
    y_integers = np.argmax(y_train[train], axis = 1)
    class_weights = class_weight.compute_class_weight("balanced", classes = np.unique(y_integers), y = y_integers)
    d_class_weights = dict(enumerate(class_weights))

    history = model.fit(X_train[train], y_train[train],
                        validation_data = (X_train[test], y_train[test]),
                        batch_size = batch_size,
                        epochs = epochs,
                        class_weight = d_class_weights,
                        verbose = 0)

    histories.append(history)

  save_model(model, os.path.join(save_dir, mod + "_cv"))
  dump_to_file(os.path.join(save_dir, mod + "_cv"), "scaler", scaler)

  return histories, model, X_val, y_val

def translate_labels(categorical_labels):
  translated_labels = []
  for label in categorical_labels:
    for trans_label in TRANSLATIONS:
      if label in TRANSLATIONS[trans_label]:
        translated_labels.append(trans_label)

  return np.array(translated_labels)
