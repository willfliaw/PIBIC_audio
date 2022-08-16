# utils/feature_selection.py

import os
import pickle
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectKBest, chi2, f_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils import class_weight

from utils.config import (DATA_PATH, DATABASES, DESIRED_FEATURES_PATH,
                          MODELS_PATH, OUTPUTS, SELECTIONS)

def dump_to_file(path, name, obj):
  with open(os.path.join(path, name + ".pkl"), "wb") as f:
    pickle.dump(obj, f)

def load_from_file(path):
  with open(path, "rb") as f:
    return pickle.load(f)

def univariate_selection(database, n_select, sel, X_train, cols, y_train):
  if sel == "f_classif":
    X = SelectKBest(score_func = f_classif, k = n_select).fit(X = X_train, y = y_train)
  elif sel == "chi2":
    X = SelectKBest(score_func = chi2, k = n_select).fit(X = X_train, y = y_train)

  dump_to_file(os.path.join(DESIRED_FEATURES_PATH, database), sel, [feat for boole, feat in zip(X.get_support(), cols) if boole])

def rfe(database, n_select, sel, X_train, cols, y_train, y_test):
  if sel == "rfecv_rf":
    grid_search = GridSearchCV(
      estimator = RandomForestClassifier(
        n_estimators = 500,
        n_jobs = -1,
        random_state = 0,
        class_weight = "balanced"
      ),
      param_grid = {
        "max_depth": [None] + [int(x) for x in np.linspace(10, 100, num = 10)],
        "max_features": ["log2", "sqrt"],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 3, 4, 5]
      },
      cv = 5,
      scoring = "f1_weighted",
      n_jobs = -1,
      verbose = 1
    ).fit(X = X_train, y = y_train)
    estimator = RandomForestClassifier(
      n_estimators = 500,
      n_jobs = -1,
      random_state = 0,
      class_weight = "balanced",
      **grid_search.best_params_
    )
  elif sel == "rfecv_svc":
    grid_search = GridSearchCV(
      estimator = SVC(
        kernel = "linear",
        random_state = 0,
        class_weight = "balanced"
      ),
      param_grid = {
        "C": [0.1, 1, 10, 100, 1000]
      },
      cv = 5,
      scoring = "f1_weighted",
      n_jobs = -1,
      verbose = 1
    ).fit(X = X_train, y = y_train)
    estimator = SVC(
      kernel = "linear",
      random_state = 0,
      class_weight = "balanced",
      **grid_search.best_params_
    )
  elif sel == "rfecv_et":
    grid_search = GridSearchCV(
      estimator = ExtraTreesClassifier(
        n_estimators = 500,
        n_jobs = -1,
        random_state = 0,
        class_weight = "balanced"
      ),
      param_grid = {
        "max_depth": [None] + [int(x) for x in np.linspace(10, 100, num = 10)],
        "max_features": ["log2", "sqrt"],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 3, 4, 5]
      },
      cv = 5,
      scoring = "f1_weighted",
      n_jobs = -1,
      verbose = 1
    ).fit(X = X_train, y = y_train)
    estimator = ExtraTreesClassifier(
      n_estimators = 500,
      n_jobs = -1,
      random_state = 0,
      class_weight = "balanced",
      **grid_search.best_params_
    )
  X_rfecv = RFECV(
    estimator = estimator,
    cv = 5,
    step = 1,
    scoring = "f1_weighted",
    min_features_to_select = n_select,
    verbose = 1
  ).fit(X = X_train, y = y_train)

  dump_to_file(os.path.join(DESIRED_FEATURES_PATH, database), sel, [feat for boole, feat in zip(X_rfecv.get_support(), cols) if boole])

def pca(database, n_select, X_train):
  pca = PCA(n_components = n_select).fit(X_train)
  for db in DATABASES:
    X_pca = pd.DataFrame(pca.transform(pd.read_csv(os.path.join(DATA_PATH, "general", "features", "1d", db + "_features.csv"), index_col = 0)))
    X_pca.to_csv(os.path.join(DESIRED_FEATURES_PATH, database, db + "_pca.csv"))

def get_f1s(database, mod):
  f1s = {}
  for sel in os.listdir(os.path.join(MODELS_PATH, database)):
    if sel[-3:] == "pca":
      f1s[sel + ".csv"] = np.round(load_from_file(os.path.join(MODELS_PATH, database, sel, mod, "f1.pkl"))*100, 2)
    else:
      f1s[sel + ".pkl"] = np.round(load_from_file(os.path.join(MODELS_PATH, database, sel, mod, "f1.pkl"))*100, 2)
  return f1s

def get_best_sel(database, mod):
  f1s = get_f1s(database, mod)
  return list(f1s.keys())[np.argmax(list(f1s.values()))]
