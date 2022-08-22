# utils/config.py

import os

BATCH_SIZE = 64
EPOCHS = 6000
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
DATABASES = ["alumni", "emodb", "ravdess_songs", "ravdess_speeches", "soundtracks"]
DESIRED_FEATURES_PATH = os.path.join(DATA_PATH, "general", "features", "1d", "desired")
FINETUNED_MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "finetunedModels"))
HOP_LENGTH = 512
MODELS_HUB = ["superb/wav2vec2-base-superb-er", "superb/wav2vec2-large-superb-er", "superb/hubert-base-superb-er", "superb/hubert-large-superb-er"]
MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "models"))
N_FFT = 2048
N_MELS = 128
N_MFCC = 13
N_SPLITS = 5
OUTPUTS = { # <database>: {<target_column>: [<desired_classes>]}}
  "emodb": {"Emotion": ["Anger", "Anxiety/Fear", "Boredom", "Disgust", "Happiness", "Neutral", "Sadness"]},
  "ravdess_songs": {"Emotion": ["angry", "calm", "fearful", "happy", "neutral", "sad"]},
  "ravdess_speeches": {"Emotion": ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]},
  "soundtracks": {"TARGET": ["ANGER", "FEAR", "HAPPY", "SAD", "SURPRISE", "TENDER"]}
}
PRETRAINED_MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "pretrainedModels"))
SEGMENT_DURATION = 3
SELECTIONS = ["f_classif", "chi2", "rfecv_rf", "rfecv_svc", "rfecv_et", "pca"]
SR = 44100
TRANSLATIONS = {
  "positive": ["Happiness", "happy", "surprised", "HAPPY", "SURPRISE", "hap"],
  "neutral": ["Boredom", "Neutral", "calm", "neutral", "TENDER", "neu"],
  "negative": ["Anger", "Anxiety/Fear", "Disgust", "Sadness", "angry", "disgust", "fearful", "sad", "ANGER", "FEAR", "SAD", "ang"]
}
WINDOW = "hann"
