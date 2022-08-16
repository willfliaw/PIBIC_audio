# utils/feature_extraction.py

import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import HOP_LENGTH, N_FFT, N_MELS, N_MFCC, SR, WINDOW

def extract_features1d(data, sr = SR, n_fft = N_FFT, hop_length = HOP_LENGTH, n_mfcc = N_MFCC, n_mels = N_MELS, window = WINDOW):
  features = {}
  mfccs = [[], [], [], [], [], [], [], [], [], [], [], [], []]
  deltas = [[], [], [], [], [], [], [], [], [], [], [], [], []]
  d_deltas = [[], [], [], [], [], [], [], [], [], [], [], [], []]
  chromas = [[], [], [], [], [], [], [], [], [], [], [], []]
  zcrs = []
  spec_cents = []
  spec_rolls = []
  rmses = []
  energies = []
  f0s = []
  tessitures = []

  for i in tqdm(range(len(data)), ncols = 50):
    audio = data[i]

    f0, _, _ = librosa.pyin(audio, fmin = librosa.note_to_hz("C2"), fmax = librosa.note_to_hz("C7"), sr = sr, hop_length = hop_length)
    tessiture = np.nanmax(f0) - np.nanmin(f0)
    # audio = librosa.effects.preemphasis(audio, coef = 0.97)
    mfcc = librosa.feature.mfcc(audio, sr = sr, n_fft = n_fft, hop_length = hop_length, window = window, n_mfcc = n_mfcc)
    delta =  librosa.feature.delta(mfcc, order = 1)
    d_delta =  librosa.feature.delta(mfcc, order = 2)
    chroma = librosa.feature.chroma_stft(audio, sr = sr, n_fft = n_fft, hop_length = hop_length, window = window)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length = hop_length)[0]
    spec_cent = librosa.feature.spectral_centroid(audio, sr = sr, n_fft = n_fft, hop_length = hop_length, window = window)[0]
    spec_roll = librosa.feature.spectral_rolloff(audio, sr = sr, n_fft = n_fft, hop_length = hop_length, window = window)[0]
    rmse = librosa.feature.rms(audio, frame_length = n_fft, hop_length = hop_length)[0]
    energy = np.mean(librosa.util.frame(audio, frame_length = n_fft, hop_length = hop_length, axis = 0)**2, axis = -1)

    for j in range(n_mfcc):
      mfccs[j].append(mfcc[j])
      deltas[j].append(delta[j])
      d_deltas[j].append(d_delta[j])
    for j in range(len(chromas)):
      chromas[j].append(chroma[j])
    zcrs.append(zcr)
    spec_cents.append(spec_cent)
    spec_rolls.append(spec_roll)
    rmses.append(rmse)
    energies.append(energy)
    f0s.append(f0)
    tessitures.append(tessiture)

  for i in range(n_mfcc):
    features["mfcc" + str(i + 1)] = mfccs[i]
    features["delta" + str(i + 1)] = deltas[i]
    features["d_delta" + str(i + 1)] = d_deltas[i]
  for i in range(len(chromas)):
    features["chroma" + str(i + 1)] = chromas[i]
  features["zcr"] = zcrs
  features["spec_cent"] = spec_cents
  features["spec_roll"] = spec_rolls
  features["rmse"] = rmses
  features["energy"] = energies
  features["f0"] = f0s
  features["tessiture"] = tessiture

  return pd.DataFrame(features)

def compute_hsf(features_vect):
  features_hsf = {}
  for i in features_vect:
    if i != "tessiture":
      feature_mean = []
      feature_var = []
      feature_max = []
      for j in features_vect[i]:
        feature_mean.append(np.nanmean(j))
        feature_var.append(np.nanvar(j))
        feature_max.append(np.nanmax(j))
      features_hsf[i + str("_mean")] = feature_mean
      features_hsf[i + str("_var")] = feature_var
      features_hsf[i + str("_max")] = feature_max
    else:
        features_hsf[i] = features_vect[i]

  return pd.DataFrame(features_hsf)

def extract_features2d(data, save_path, sr = SR, n_fft = N_FFT, hop_length = HOP_LENGTH, n_mfcc = N_MFCC, n_mels = N_MELS, window = WINDOW):
  os.makedirs(save_path, exist_ok = True)
  for i in tqdm(range(len(data)), ncols = 50):
    audio = data[i]
    melgram = librosa.power_to_db(librosa.feature.melspectrogram(audio, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, window = window), ref = np.max)
    np.save(os.path.join(save_path, str(i) + ".npy"), melgram)
