# select_features.py

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import utils

def features1d_selection(database, n_select, *selection):
  assert (database in utils.DATABASES or database == "concat") and database != "alumni"
  os.makedirs(os.path.join(utils.DESIRED_FEATURES_PATH, database), exist_ok = True)

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

  X_train, X_test, y_train, y_test = train_test_split(X, categorical_y, stratify = categorical_y, test_size = 0.4, random_state = 0)
  scaler = MinMaxScaler()
  X_train_std = scaler.fit_transform(X_train)

  for sel in selection:
    assert sel in utils.SELECTIONS
    if sel == "f_classif" or sel == "chi2":
      utils.univariate_selection(database, n_select, sel, X_train_std, X.columns, y_train)
    elif sel == "rfecv_rf" or sel == "rfecv_svc" or sel == "rfecv_et":
      utils.rfe(database, n_select, sel, X_train_std, X.columns, y_train, y_test)
    elif sel == "pca":
      utils.pca(database, n_select, X_train)

def eval_selected_features(database, *mods):
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

  for sel in tqdm(os.listdir(os.path.join(utils.DESIRED_FEATURES_PATH, database)), ncols = 50):
    if sel[-3:] == "pkl":
      X_sel = X[(feat for feat in X if feat in utils.load_from_file(os.path.join(utils.DESIRED_FEATURES_PATH, database, sel)))]
    elif sel == database + "_pca.csv":
      X_sel = pd.read_csv(os.path.join(utils.DESIRED_FEATURES_PATH, database, sel), index_col = 0)[pd.Series(info[next(iter(utils.OUTPUTS[database]))][i] in utils.OUTPUTS[database][next(iter(utils.OUTPUTS[database]))] for i, _ in info.iterrows())]
    if sel[-3:] == "pkl" or sel == database + "_pca.csv":
      for mod in mods:
        os.makedirs(os.path.join(utils.MODELS_PATH, database, sel[:-4]), exist_ok = True)
        history, model, X_val, y_val = utils.train_model(mod, X_sel, y, os.path.join(utils.MODELS_PATH, database, sel[:-4]))
        utils.model_eval(history, np.argmax(y_val, axis = 1), np.argmax(model.predict(X_val), axis = 1),list(utils.TRANSLATIONS.keys()), os.path.join(utils.MODELS_PATH, database, sel[:-4], mod))

if __name__ == "__main__":
  for db in utils.OUTPUTS:
    features1d_selection(db, 13, "rfecv_rf", "rfecv_svc", "rfecv_et")
    nums = [len(utils.load_from_file(os.path.join(utils.DESIRED_FEATURES_PATH, db, sel))) for sel in ["rfecv_rf.pkl", "rfecv_svc.pkl", "rfecv_et.pkl"]]
    features1d_selection(db, 5*(min(nums)//5), "f_classif", "chi2", "pca")
    eval_selected_features(db, "dnn")
  features1d_selection("concat", 13, "rfecv_rf", "rfecv_svc", "rfecv_et")
  nums = [len(utils.load_from_file(os.path.join(utils.DESIRED_FEATURES_PATH, "concat", sel))) for sel in ["rfecv_rf.pkl", "rfecv_svc.pkl", "rfecv_et.pkl"]]
  features1d_selection("concat", 5*(min(nums)//5), "f_classif", "chi2", "pca")
  eval_selected_features("concat", "dnn")
