# train_models.py

import os

import numpy as np
import pandas as pd

import utils

def train_plot_model(database, sel, mod, epochs):
  assert (database in utils.DATABASES or database == "concat") and database != "alumni"
  if database != "concat":
    features = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "features", "1d", database + "_features.csv"), index_col = 0)
    info = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "info", database + "_info.csv"), index_col = 0)
    data = features.join(info[next(iter(utils.OUTPUTS[database]))]).dropna()
    data = data[pd.Series(data[next(iter(utils.OUTPUTS[database]))][i] in utils.OUTPUTS[database][next(iter(utils.OUTPUTS[database]))] for i, _ in data.iterrows())]
    X = data.drop([next(iter(utils.OUTPUTS[database]))], axis = 1)
    categorical_y = utils.translate_labels(data[next(iter(utils.OUTPUTS[database]))])
  else:
    X_concat = []
    categorical_y_concat = []
    for db in utils.OUTPUTS:
      features = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "features", "1d", db + "_features.csv"), index_col = 0)
      info = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "info", db + "_info.csv"), index_col = 0)
      data = features.join(info[next(iter(utils.OUTPUTS[db]))]).dropna()
      data = data[pd.Series(data[next(iter(utils.OUTPUTS[db]))][i] in utils.OUTPUTS[db][next(iter(utils.OUTPUTS[db]))] for i, _ in data.iterrows())]
      X_db = data.drop([next(iter(utils.OUTPUTS[db]))], axis = 1)
      categorical_y_db = pd.DataFrame(utils.translate_labels(data[next(iter(utils.OUTPUTS[db]))]))
      X_concat.append(X_db)
      categorical_y_concat.append(categorical_y_db)
    X = pd.concat(X_concat)
    X = X.reset_index(drop = True)
    categorical_y = pd.concat(categorical_y_concat)
    categorical_y = np.ravel(categorical_y.reset_index(drop = True))

  y = np.array([np.array([1. if i == list(utils.TRANSLATIONS.keys())[j] else 0. for j in range(len(utils.TRANSLATIONS))]) for i in categorical_y])


  if sel[-3:] == "pkl":
    X_sel = X[(feat for feat in X if feat in utils.load_from_file(os.path.join(utils.DESIRED_FEATURES_PATH, database, sel)))]
  elif sel == database + "_pca.csv":
    X_sel = pd.read_csv(os.path.join(utils.DESIRED_FEATURES_PATH, database, sel), index_col = 0)

  histories, model, X_val, y_val = utils.train_model_cv(mod, X_sel, y, os.path.join(utils.MODELS_PATH, database, sel[:-4]), epochs = epochs)
  utils.model_eval_cv(histories, np.argmax(y_val, axis = 1), np.argmax(model.predict(X_val), axis = 1), list(utils.TRANSLATIONS.keys()), os.path.join(utils.MODELS_PATH, database, sel[:-4], mod + "_cv"))

if __name__ == "__main__":
  for db in list(utils.OUTPUTS.keys()):
    train_plot_model(db, utils.get_best_sel(db, "dnn"), "dnn", 550)
    print("F1 dnn:", np.round(utils.load_from_file(os.path.join(utils.MODELS_PATH, db, utils.get_best_sel(db, "dnn")[:-4], "dnn", "f1.pkl"))*100, 2))
    print("F1 dnn_cv:", np.round(utils.load_from_file(os.path.join(utils.MODELS_PATH, db, utils.get_best_sel(db, "dnn")[:-4], "dnn_cv", "f1.pkl"))*100, 2))
  train_plot_model("concat", utils.get_best_sel("concat", "dnn"), "dnn", 550)
  print("F1 dnn:", np.round(utils.load_from_file(os.path.join(utils.MODELS_PATH, "concat", utils.get_best_sel("concat", "dnn")[:-4], "dnn", "f1.pkl"))*100, 2))
  print("F1 dnn_cv:", np.round(utils.load_from_file(os.path.join(utils.MODELS_PATH, "concat", utils.get_best_sel("concat", "dnn")[:-4], "dnn_cv", "f1.pkl"))*100, 2))
