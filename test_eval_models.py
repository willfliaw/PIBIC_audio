# test_eval_models.py

import os

import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.keras.models import load_model
from tqdm import tqdm

import utils

def test_model(model_database, sel, mod, test_database):
  assert (model_database in utils.DATABASES or model_database == "concat") and model_database != "alumni" and test_database in utils.DATABASES
  test_features = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "features", "1d", test_database + "_features.csv"), index_col = 0)
  if test_database == "alumni":
    X = test_features.dropna()
  else:
    test_info = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "info", test_database + "_info.csv"), index_col = 0)
    test_data = test_features.join(test_info[next(iter(utils.OUTPUTS[test_database]))]).dropna()
    test_data = test_data[pd.Series(test_data[next(iter(utils.OUTPUTS[test_database]))][i] in utils.OUTPUTS[test_database][next(iter(utils.OUTPUTS[test_database]))] for i, _ in test_data.iterrows())]
    X = test_data.drop([next(iter(utils.OUTPUTS[test_database]))], axis = 1)

  if sel[-3:] == "pkl":
    X_sel = X[(feat for feat in X if feat in utils.load_from_file(os.path.join(utils.DESIRED_FEATURES_PATH, model_database, sel)))]
  elif sel == test_database + "_pca.csv":
    X_sel = pd.read_csv(os.path.join(utils.DESIRED_FEATURES_PATH, model_database, sel), index_col = 0)
    if test_database != "alumni":
      X_sel = X_sel[pd.Series(test_info[next(iter(utils.OUTPUTS[database]))][i] in utils.OUTPUTS[database][next(iter(utils.OUTPUTS[database]))] for i, _ in test_info.iterrows())]

  os.makedirs(os.path.join(utils.MODELS_PATH, model_database, sel[:-4], mod, "preds"), exist_ok = True)
  model = load_model(os.path.join(utils.MODELS_PATH, model_database, sel[:-4], mod), compile = True)
  scaler = utils.load_from_file(os.path.join(utils.MODELS_PATH, model_database, sel[:-4], mod, "scaler.pkl"))
  pred = model.predict(scaler.transform(X_sel))

  if test_database == "alumni":
    pred_df = pd.DataFrame(pred)
    pred_df.columns = utils.TRANSLATIONS
    pred_df.to_csv(os.path.join(utils.MODELS_PATH, model_database, sel[:-4], mod, "preds", test_database + ".csv"))

  categorical_pred_df = pd.DataFrame([list(utils.TRANSLATIONS.keys())[i] for i in np.argmax(pred, axis = 1)])
  categorical_pred_df.columns = ["Prediction"]
  if test_database != "alumni":
    test_info = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "info", test_database + "_info.csv"), index_col = 0).dropna()
    test_info = test_info[pd.Series(test_info[next(iter(utils.OUTPUTS[test_database]))][i] in utils.OUTPUTS[test_database][next(iter(utils.OUTPUTS[test_database]))] for i, _ in test_info.iterrows())]
    test_categorical_y = pd.DataFrame(utils.translate_labels(test_info[next(iter(utils.OUTPUTS[test_database]))]), columns = [next(iter(utils.OUTPUTS[test_database]))])
    categorical_pred_df = categorical_pred_df.join(test_categorical_y)
  categorical_pred_df.to_csv(os.path.join(utils.MODELS_PATH, model_database, sel[:-4], mod, "preds", test_database + "_categorical.csv"))

  return pred

def eval_model(model_database, sel, mod, predictions, test_database):
  if test_database == "alumni":
    categorical_predictions = pd.read_csv(os.path.join(utils.MODELS_PATH, model_database, sel[:-4], "dnn_cv", "preds", "alumni.csv"), index_col = 0).dropna()
    utils.plot_alumni(model_database, sel, mod, predictions)
  else:
    test_info = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "info", test_database + "_info.csv"), index_col = 0).dropna()
    test_info = test_info[pd.Series(test_info[next(iter(utils.OUTPUTS[test_database]))][i] in utils.OUTPUTS[test_database][next(iter(utils.OUTPUTS[test_database]))] for i, _ in test_info.iterrows())]
    test_categorical_y = utils.translate_labels(test_info[next(iter(utils.OUTPUTS[test_database]))])
    test_y = np.array([np.array([1. if i == list(utils.TRANSLATIONS.keys())[j] else 0. for j in range(len(utils.TRANSLATIONS))]) for i in test_categorical_y])
    utils.dump_to_file(os.path.join(utils.MODELS_PATH, model_database, sel[:-4], mod, "preds", "f1s"), test_database + "_f1", metrics.f1_score(np.argmax(test_y, axis = 1), np.argmax(predictions, axis = 1), average = "weighted"))
    utils.plot_confusion_matrix(np.argmax(test_y, axis = 1), np.argmax(predictions, axis = 1), list(utils.TRANSLATIONS.keys()), os.path.join(utils.MODELS_PATH, model_database, sel[:-4], mod, "preds", "matrices"), name = test_database)

if __name__ == "__main__":
  for mod_db in tqdm(utils.OUTPUTS, ncols = 50):
    os.makedirs(os.path.join(utils.MODELS_PATH, mod_db, utils.get_best_sel(mod_db, "dnn")[:-4], "dnn_cv", "preds", "matrices"), exist_ok = True)
    os.makedirs(os.path.join(utils.MODELS_PATH, mod_db, utils.get_best_sel(mod_db, "dnn")[:-4], "dnn_cv", "preds", "f1s"), exist_ok = True)
    os.makedirs(os.path.join(utils.MODELS_PATH, mod_db, utils.get_best_sel(mod_db, "dnn")[:-4], "dnn_cv", "preds", "alumni_eval"), exist_ok = True)
    for test_db in tqdm(utils.DATABASES, ncols = 50):
      if mod_db != test_db:
        pred = test_model(mod_db, utils.get_best_sel(mod_db, "dnn"), "dnn_cv", test_db)
        eval_model(mod_db, utils.get_best_sel(mod_db, "dnn"), "dnn_cv", pred, test_db)
  os.makedirs(os.path.join(utils.MODELS_PATH, "concat", utils.get_best_sel("concat", "dnn")[:-4], "dnn_cv", "preds", "alumni_eval"), exist_ok = True)
  pred = test_model("concat", utils.get_best_sel("concat", "dnn"), "dnn_cv", "alumni")
  eval_model("concat", utils.get_best_sel("concat", "dnn"), "dnn_cv", pred, "alumni")
