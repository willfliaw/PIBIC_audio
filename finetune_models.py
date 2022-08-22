# finetune_models.py

import os

import numpy as np
import pandas as pd
from transformers import Wav2Vec2FeatureExtractor

import utils

def finetune_plot_model(hub, database, batch_size = 16, epochs = 3):
  assert database in utils.DATABASES + ["all"] and database != "alumni"
  if database != "all":
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hub, return_attention_mask = False)
    _, audios = utils.read_datasets(database, feature_extractor.sampling_rate)
    feature_space = pd.DataFrame({"input_values": list(feature_extractor(
      audios,
      sampling_rate = feature_extractor.sampling_rate,
      max_length = int(feature_extractor.sampling_rate*utils.SEGMENT_DURATION),
      truncation = True,
      padding = True,
      return_tensors = "pt",
    )["input_values"])})
    info = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "info", database + "_info.csv"), index_col = 0).dropna()
    data = feature_space.join(info[next(iter(utils.OUTPUTS[database]))]).dropna()
    data = data[pd.Series(data[next(iter(utils.OUTPUTS[database]))][i] in utils.OUTPUTS[database][next(iter(utils.OUTPUTS[database]))] for i, _ in data.iterrows())]
    X = data.drop([next(iter(utils.OUTPUTS[database]))], axis = 1)
    categorical_y = utils.translate_labels(info[next(iter(utils.OUTPUTS[database]))])
  else:
    X_concat, categorical_y_concat = list(), list()
    for db in utils.OUTPUTS:
      feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hub, return_attention_mask = False)
      _, audios = utils.read_datasets(db, feature_extractor.sampling_rate)
      feature_space = pd.DataFrame({"input_values": feature_extractor(
        audios,
        sampling_rate = feature_extractor.sampling_rate,
        max_length = int(feature_extractor.sampling_rate*utils.SEGMENT_DURATION),
        truncation = True
      )["input_values"]})
      info = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "info", db + "_info.csv"), index_col = 0)
      data = feature_space.join(info[next(iter(utils.OUTPUTS[db]))]).dropna()
      data = data[pd.Series(data[next(iter(utils.OUTPUTS[db]))][i] in utils.OUTPUTS[db][next(iter(utils.OUTPUTS[db]))] for i, _ in data.iterrows())]
      X_db = data.drop([next(iter(utils.OUTPUTS[db]))], axis = 1)
      categorical_y_db = pd.DataFrame(utils.translate_labels(data[next(iter(utils.OUTPUTS[db]))]))
      X_concat.append(X_db)
      categorical_y_concat.append(categorical_y_db)
    X = pd.concat(X_concat)
    X = X.reset_index(drop = True)
    categorical_y = pd.concat(categorical_y_concat)
    categorical_y = np.ravel(categorical_y.reset_index(drop = True))

  y = np.array([list(utils.TRANSLATIONS.keys()).index(i) for i in categorical_y])

  label2id, id2label = dict(), dict()
  for i, label in enumerate(utils.TRANSLATIONS.keys()):
    label2id[label] = str(i)
    id2label[str(i)] = label

  history, model, X_val, y_val = utils.train_model(hub.split("/")[-1].split("-")[0], X, y, os.path.join(utils.FINETUNED_MODELS_PATH, database), batch_size = batch_size, epochs = epochs, hub = hub, feature_extractor = feature_extractor, label2id = label2id, id2label = id2label)
  utils.model_eval(history, np.argmax(y_val, axis = 1), np.argmax(model.predict(X_val), axis = 1), list(utils.TRANSLATIONS.keys()), os.path.join(utils.FINETUNED_MODELS_PATH, database))

if __name__ == "__main__":
  # for hub in utils.MODELS_HUB:
  #   for db in utils.DATABASES:
  #     history, model, X_val, y_val = finetune_plot_model(hub, db)
  #     utils.model_eval(history, np.argmax(y_val, axis = 1), np.argmax(model.predict(X_val), axis = 1), list(utils.TRANSLATIONS.keys()), os.path.join(utils.MODELS_PATH, db))
  hub = "superb/wav2vec2-base-superb-er"
  db = "emodb"
  history, model, X_val, y_val = finetune_plot_model(hub, db)
  utils.model_eval(history, np.argmax(y_val, axis = 1), np.argmax(model.predict(X_val), axis = 1), list(utils.TRANSLATIONS.keys()), os.path.join(utils.MODELS_PATH, db))

