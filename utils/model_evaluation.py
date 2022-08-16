# utils/model_evaluation.py

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

from utils.config import DATA_PATH, MODELS_PATH, OUTPUTS, TRANSLATIONS
from utils.feature_selection import dump_to_file

plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.family"] = "DeJavu Serif"

def plot_learning_curve(history, save_dir, name = "line"):
  fig, ax = plt.subplots(1, 2, figsize = (15, 5))

  epoch_range = range(1, pd.DataFrame(history.history).shape[0] + 1)
  ticks = np.linspace(1, pd.DataFrame(history.history).shape[0], num = 5, dtype = int, endpoint = True)

  ax[0].set_ylim(0, 1)
  ax[0].set_xticks(ticks)
  ax[0].plot(epoch_range, history.history["accuracy"])
  ax[0].plot(epoch_range, history.history["val_accuracy"])
  ax[0].axvline(x = history.history["val_loss"].index(min(history.history["val_loss"])) + 1, color = "k", label = "Best epoch", linestyle = "--", alpha = 0.8)
  ax[0].set_ylabel("Accuracy")
  ax[0].set_xlabel("Epoch")
  ax[0].legend(["Treino", "Teste"], loc = "best")

  ax[1].set_xticks(ticks)
  ax[1].plot(epoch_range, history.history["loss"])
  ax[1].plot(epoch_range, history.history["val_loss"])
  ax[1].axvline(x = history.history["val_loss"].index(min(history.history["val_loss"])) + 1, color = "k", label = "Best epoch", linestyle = "--", alpha = 0.8)
  ax[1].set_ylabel("Loss")
  ax[1].set_xlabel("Epoch")
  ax[1].legend(["Treino", "Teste"], loc = "best")

  plt.savefig(os.path.join(save_dir, name + ".png"))
  plt.close(fig)

def plot_confusion_matrix(labels, predictions, class_names, save_dir, name = "matrix"):
  cm = tf.math.confusion_matrix(labels = labels, predictions = predictions)

  fig, ax = plt.subplots(figsize = (6, 6))
  plt.imshow(cm, interpolation = "nearest", cmap = plt.cm.Blues)
  plt.title("Matriz de confusão")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation = 30, ha = "right")
  plt.yticks(tick_marks, class_names)

  nums = np.around(cm.numpy().astype("float")*100/cm.numpy().sum(axis = 1)[:, np.newaxis], decimals = 2)

  threshold = cm.numpy().max()/2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, "\n" + str(int(cm[i, j])) + "\n" + str(nums[i, j]) + "%", horizontalalignment = "center", color = color)

  plt.tight_layout()
  plt.ylabel("Classe verdadeira")
  plt.xlabel("Classe identificada")

  plt.savefig(os.path.join(save_dir, name + ".png"))
  plt.close(fig)

def model_eval(history, labels, predictions, class_names, save_dir):
  dump_to_file(save_dir, "f1", metrics.f1_score(labels, predictions, average = "weighted"))
  plot_learning_curve(history, save_dir)
  plot_confusion_matrix(labels, predictions, class_names, save_dir)

def model_eval_cv(histories, labels, predictions, class_names, save_dir):
  dump_to_file(save_dir, "f1", metrics.f1_score(labels, predictions, average = "weighted"))
  for i in range(len(histories)):
    plot_learning_curve(histories[i], save_dir, "line" + str(i))
  plot_confusion_matrix(labels, predictions, class_names, save_dir)

def plot_alumni(model_database, sel, mod, predictions):
  info = pd.read_csv(os.path.join(DATA_PATH, "general", "info", "alumni" + "_info.csv"), index_col = 0).dropna()
  offset = 0
  for name in info["File Name"].unique():
    name_prediction = predictions[info["File Name"] == name]
    fig, ax = plt.subplots(figsize = (40, 5))
    line = np.argmax(predictions, axis = 1)[offset:offset + len(info[info["File Name"] == name])]
    indexes = np.linspace(start = 0, stop = len(info[info["File Name"] == name]), num = len(info[info["File Name"] == name]))
    plt.plot(indexes, line)
    plt.xticks(range(len(info[info["File Name"] == name]))[::len(info[info["File Name"] == name])//10])
    ax.set_xticklabels(info[info["File Name"] == name]["Location"][::len(info[info["File Name"] == name])//10])
    plt.yticks(range(0, len(TRANSLATIONS)))
    ax.set_yticklabels(list(TRANSLATIONS.keys()))
    plt.grid(b = True, axis = "y")
    plt.title(name)

    plt.savefig(os.path.join(MODELS_PATH, model_database, sel[:-4], "dnn_cv", "preds", "alumni_eval", name + "_argmax.png"))
    plt.close(fig)

    offset += len(info[info['File Name'] == name])

  plt.style.use(["science", "no-latex"])
  plt.rcParams["figure.dpi"] = 100
  plt.rcParams["font.family"] = "DeJavu Serif"

  offset = 0
  for name in info["File Name"].unique():
    name_prediction = predictions[info["File Name"] == name]
    fig, ax = plt.subplots(figsize = (40, 5))
    for i in range(len(list(TRANSLATIONS.keys()))):
      line = [sample[i] for sample in predictions][offset:offset + len(info[info["File Name"] == name])]
      indexes = np.linspace(start = 0, stop = len(info[info["File Name"] == name]), num = len(info[info["File Name"] == name]))
      plt.plot(indexes, line)
    plt.legend(list(TRANSLATIONS.keys()))
    plt.xticks(range(len(info[info["File Name"] == name]))[::len(info[info["File Name"] == name])//10])
    ax.set_xticklabels(info[info["File Name"] == name]["Location"][::len(info[info["File Name"] == name])//10])
    plt.grid(b = True, axis = "y")
    plt.title(name)

    plt.savefig(os.path.join(MODELS_PATH, model_database, sel[:-4], "dnn_cv", "preds", "alumni_eval", name + ".png"))
    plt.close(fig)

    offset += len(info[info['File Name'] == name])
