# pretrained_models.py

import os

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from tqdm import tqdm
from transformers import (HubertForSequenceClassification,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForSequenceClassification)

import utils

def test_pretrained(hub, database):
  assert database in utils.DATABASES + ["all"]
  feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hub, return_attention_mask = False)
  _, audios = utils.read_datasets(database, feature_extractor.sampling_rate)

  if hub.split("/")[-1].split("-")[0] == "wav2vec2":
    model = Wav2Vec2ForSequenceClassification.from_pretrained(hub)
  elif hub.split("/")[-1].split("-")[0] == "hubert":
    model = HubertForSequenceClassification.from_pretrained(hub)

  labels = list()
  for i in tqdm(range(len(audios)//50 + 1), ncols = 50):
    if i < len(audios)//50:
      buffer = audios[i*50:(i + 1)*50]
    else:
      buffer = audios[i*50:]

    inputs = feature_extractor(
      buffer,
      sampling_rate = feature_extractor.sampling_rate,
      max_length = int(feature_extractor.sampling_rate*utils.SEGMENT_DURATION),
      truncation = True,
      padding = True,
      return_tensors = "pt",
    )

    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    labels += [model.config.id2label[_id] for _id in predicted_ids.tolist()]
  trans_labels = utils.translate_labels(labels)
  os.makedirs(os.path.join(utils.PRETRAINED_MODELS_PATH, hub.split("/")[-1].split("-")[0], hub.split("/")[-1].split("-")[1], database), exist_ok = True)
  utils.dump_to_file(os.path.join(utils.PRETRAINED_MODELS_PATH, hub.split("/")[-1].split("-")[0], hub.split("/")[-1].split("-")[1], database), "preds", trans_labels)

  return trans_labels

def eval_pretrained_model(hub, categorical_predictions, database):
  predictions = pd.DataFrame({"predictions": [list(utils.TRANSLATIONS.keys()).index(i) for i in categorical_predictions]})
  if database != "alumni":
    test_info = pd.read_csv(os.path.join(utils.DATA_PATH, "general", "info", database + "_info.csv"), index_col = 0).dropna()
    data = predictions.join(test_info)
    data = data[pd.Series(data[next(iter(utils.OUTPUTS[database]))][i] in utils.OUTPUTS[database][next(iter(utils.OUTPUTS[database]))] for i, _ in data.iterrows())]
    test_categorical_y = utils.translate_labels(data[next(iter(utils.OUTPUTS[database]))])
    test_y = np.array([np.array([1. if i == list(utils.TRANSLATIONS.keys())[j] else 0. for j in range(len(utils.TRANSLATIONS))]) for i in test_categorical_y])
    utils.dump_to_file(os.path.join(utils.PRETRAINED_MODELS_PATH, hub.split("/")[-1].split("-")[0], hub.split("/")[-1].split("-")[1], database), "f1", metrics.f1_score(np.argmax(test_y, axis = 1), data["predictions"], average = "weighted"))
    utils.plot_confusion_matrix(np.argmax(test_y, axis = 1), data["predictions"], list(utils.TRANSLATIONS.keys()), os.path.join(utils.PRETRAINED_MODELS_PATH, hub.split("/")[-1].split("-")[0], hub.split("/")[-1].split("-")[1], database), name = database)

if __name__ == "__main__":
  # for hub in utils.MODELS_HUB:
  #   for db in utils.OUTPUTS:
  #     trans_labels = test_pretrained(hub, db)
  #     eval_pretrained_model(hub, trans_labels, db)
  hub = "superb/hubert-base-superb-er"
  db = "emodb"
  trans_labels = test_pretrained(hub, db)
  trans_labels = utils.load_from_file(os.path.join(utils.PRETRAINED_MODELS_PATH, hub.split("/")[-1].split("-")[0], hub.split("/")[-1].split("-")[1], db, "preds.pkl"))
  eval_pretrained_model(hub, trans_labels, db)
