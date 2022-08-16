# preprocess_data.py

import os
import sys

import utils

def preprocessing(database = "all", write_inf = "y", write_d1 = "y", write_d2 = "y"):
  assert database in utils.DATABASES + ["all"]
  info, audio = utils.read_datasets(database)

  if write_inf == "y":
    os.makedirs(os.path.join(utils.DATA_PATH, "general", "info"), exist_ok = True)
    if database == "all":
      for db, aud, inf in zip(utils.DATABASES, audio, info):
          inf.to_csv(os.path.join(utils.DATA_PATH, "general", "info", db + "_info.csv"))
    elif database in utils.DATABASES:
      info.to_csv(os.path.join(utils.DATA_PATH, "general", "info", database + "_info.csv"))

  if write_d1 == "y":
    os.makedirs(os.path.join(utils.DATA_PATH, "general", "features", "1d"), exist_ok = True)
    if database == "all":
      for db, aud, inf in zip(utils.DATABASES, audio, info):
        utils.compute_hsf(utils.extract_features1d(aud)).to_csv(os.path.join(utils.DATA_PATH,  "general", "features", "1d", db + "_features.csv"))
    elif database in utils.DATABASES:
      utils.compute_hsf(utils.extract_features1d(audio)).to_csv(os.path.join(utils.DATA_PATH,  "general", "features", "1d", database + "_features.csv"))

  if write_d2 == "y":
    if database == "all":
      for db, aud in zip(utils.DATABASES, audio, info):
        utils.extract_features2d(aud, os.path.join(utils.DATA_PATH, "general", "features", "2d", db + "_imgs"))
    elif database in utils.DATABASES:
      utils.extract_features2d(audio, os.path.join(utils.DATA_PATH, "general", "features", "2d", database + "_imgs"))

if __name__ == "__main__":
  preprocessing()
