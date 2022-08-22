# utils/models.py

import logging
import os
import random

import numpy as np
import tensorflow as tf
import torch
from datasets import load_metric
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Dropout)
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tqdm import tqdm
from transformers import (HubertForSequenceClassification, Trainer,
                          TrainingArguments, Wav2Vec2ForSequenceClassification)

from utils.config import BATCH_SIZE, EPOCHS, MODELS_HUB, N_SPLITS, TRANSLATIONS
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

def compute_metrics(eval_pred):
  metric = load_metric("accuracy")
  predictions = np.argmax(eval_pred.predictions, axis=1)
  return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def train_model(mod, feature_space, labels, save_dir, batch_size = BATCH_SIZE, epochs = EPOCHS, hub = "", feature_extractor = None, label2id = dict(), id2label = dict()):
  X_train, X_val, y_train, y_val = train_test_split(feature_space, labels, stratify = labels, test_size = 0.2, random_state = 0)
  X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.25, random_state = 0)

  if mod not in list(set([i.split("/")[-1].split("-")[0] for i in MODELS_HUB])):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
  else:
    X_train = X_train.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    X_val = X_val.reset_index(drop = True)
    class database(torch.utils.data.Dataset):
      def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
      def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
      def __len__(self):
        return len(self.labels)
    train_dataset = database(X_train, y_train)
    test_dataset = database(X_test, y_test)
    val_dataset = database(X_val, y_val)

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
  else:
    if mod == "wav2vec2":
      model = Wav2Vec2ForSequenceClassification.from_pretrained(
        hub,
        num_labels = len(TRANSLATIONS.keys()),
        label2id = label2id,
        id2label = id2label,
        ignore_mismatched_sizes = True
      )
    elif mod == "hubert":
      model = HubertForSequenceClassification.from_pretrained(
        hub,
        num_labels = len(TRANSLATIONS.keys()),
        label2id = label2id,
        id2label = id2label,
        ignore_mismatched_sizes = True
      )
  # elif mod == "cnn1d":
  # elif mod = "cnn2d":

  if mod not in list(set([i.split("/")[-1].split("-")[0] for i in MODELS_HUB])):
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
  else:
    training_args = TrainingArguments(
      output_dir = os.path.join(save_dir, mod, hub.split("/")[-1].split("-")[1]),
      evaluation_strategy = "epoch",
      save_strategy = "epoch",
      learning_rate = 3e-5,
      per_device_train_batch_size = batch_size,
      gradient_accumulation_steps=4,
      per_device_eval_batch_size = batch_size,
      num_train_epochs = epochs,
      warmup_ratio = 0.1,
      logging_steps = 10,
      load_best_model_at_end = True,
      metric_for_best_model = "accuracy",
      push_to_hub = False,
      gradient_checkpointing = True
    )
    trainer = Trainer(
      model = model,
      args = training_args,
      train_dataset = train_dataset,
      eval_dataset = test_dataset,
      compute_metrics = compute_metrics
    )
    trainer.train()
    torch.save(model, os.path.join(save_dir, mod, hub.split("/")[-1].split("-")[1]))

  return history, model, X_val, y_val

def train_model_cv(mod, feature_space, labels, save_dir, n_splits = N_SPLITS, batch_size = BATCH_SIZE, epochs = EPOCHS, hub = ""):
  histories, acc_per_fold, loss_per_fold = list(), list(), list()

  X_train, X_val, y_train, y_val = train_test_split(feature_space, labels, stratify = labels, test_size = 0.2, random_state = 0)

  if mod not in list(set([i.split("/")[-1].split("-")[0] for i in MODELS_HUB])):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
  else:
    ...

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
  elif mod == "wav2vec2":
    ...
  elif mod == "hubert":
    ...
  # elif mod == "cnn1d":
  # elif mod = "cnn2d":

  if mod not in list(set([i.split("/")[-1].split("-")[0] for i in MODELS_HUB])):
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
  else:
    ...

  return histories, model, X_val, y_val

def translate_labels(categorical_labels):
  translated_labels = list()
  for label in categorical_labels:
    for trans_label in TRANSLATIONS:
      if label in TRANSLATIONS[trans_label]:
        translated_labels.append(trans_label)

  return np.array(translated_labels)
