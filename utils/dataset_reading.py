# utils/read_datasets.py

import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import DATA_PATH, SEGMENT_DURATION, SR

def read_alumni(sr = SR, segment_duration = SEGMENT_DURATION):
    speeches, names, locations = list(), list(), list()

    for fn in tqdm(np.asarray(os.listdir(os.path.join(DATA_PATH, "alumni", "audios"))), ncols = 50):
        offset = 0
        path = os.path.join(DATA_PATH, "alumni", "audios", fn)
        speech, sr = librosa.load(path, sr = sr, mono = True, offset = offset,  duration = None)
        S_full, phase = librosa.magphase(librosa.stft(speech))
        S_filter = librosa.decompose.nn_filter(S_full, aggregate = np.median, metric = "cosine", width = int(librosa.time_to_frames(0.75, sr = sr)))
        S_filter = np.minimum(S_full, S_filter)
        mask_v = librosa.util.softmask(S_full - S_filter, 10*S_filter, power = 2)
        speech = librosa.istft(mask_v*S_full*phase)
        indexes = librosa.effects.split(speech, top_db = 60)
        speech = np.array([num for arr in (speech[i[0]:i[1]] for i in indexes) for num in arr.tolist()])
        for i in range(int(len(speech)/sr)//segment_duration - 1):
            speech_segment = speech[offset*sr:(offset + segment_duration)*sr]
            speeches.append(speech_segment)
            names.append(fn[: -4])
            locations.append(str(offset) + " - " + str(offset + segment_duration))
            offset += segment_duration

    info = pd.DataFrame({"File Name": names})
    location = pd.DataFrame({"Location": locations})
    info = info.join(location)

    sexes = list()
    coded_sexes = {"andre": "Male",
                   "bruna": "Female",
                   "bruno": "Male",
                   "lucas": "Male",
                   "luisa": "Female",
                   "patrick": "Male",
                   "pedro": "Male",
                   "ryan": "Male"}

    for fn in range(len(info["File Name"])):
        for i in coded_sexes:
                if info["File Name"][fn] == i:
                    sexes.append(coded_sexes[i])

    sexes = pd.DataFrame({"Sex": sexes})

    info = info.join(sexes)

    return info, speeches

def read_emodb(sr = SR, pad = False, trunc = False):
  speeches, names, durations = list(), list(), list()

  for fn in tqdm(np.asarray(os.listdir(os.path.join(DATA_PATH, "EmoDB", "wav"))), ncols = 50):
    path = os.path.join(DATA_PATH, "EmoDB", "wav", fn)
    if not trunc:
      speech, sr = librosa.load(path, sr = sr, mono = True)
    else:
      speech, sr = librosa.load(path, sr = sr, mono = True, duration = SEGMENT_DURATION)
    indexes = librosa.effects.split(speech, top_db = 60)
    speech = np.array([num for arr in (speech[i[0]:i[1]] for i in indexes) for num in arr.tolist()])
    if pad and len(speech) < SEGMENT_DURATION*sr:
        speech = np.array([speech[i] if i < len(speech) else 0. for i in range(SEGMENT_DURATION*sr)])
    speeches.append(speech)
    names.append(fn[: -4])
    durations.append(len(speech)/sr)

  info = pd.DataFrame({"File Name": names})

  sexes, ages, texts, emotions = list(), list(), list(), list()

  coded_actors = {
    "03": ["Male",    31],
    "08": ["Female",  34],
    "09": ["Female",  21],
    "10": ["Male",    32],
    "11": ["Male",    26],
    "12": ["Male",    30],
    "13": ["Female",  32],
    "14": ["Female",  35],
    "15": ["Male",    25],
    "16": ["Female",  31]
  }
  coded_texts = [
    "a01",
    "a02",
    "a04",
    "a05",
    "a07",
    "b01",
    "b02",
    "b03",
    "b09",
    "b10"
  ]
  coded_emotions = {
    "W": "Anger",
    "L": "Boredom",
    "E": "Disgust",
    "A": "Anxiety/Fear",
    "F": "Happiness",
    "T": "Sadness",
    "N": "Neutral"
  }

  for fn in range(len(info["File Name"])):
    for i in coded_actors:
      if info["File Name"][fn][: 2] == i:
        sexes.append(coded_actors[i][0])
        ages.append(coded_actors[i][1])
    for i in coded_texts:
      if info["File Name"][fn][2: 5] == i:
        texts.append(i)
    for i in coded_emotions:
      if info["File Name"][fn][5] == i:
        emotions.append(coded_emotions[i])

  durations = pd.DataFrame({"Duration": durations})
  sexes = pd.DataFrame({"Sex": sexes})
  ages = pd.DataFrame({"Age": ages})
  texts = pd.DataFrame({"Text": texts})
  emotions = pd.DataFrame({"Emotion": emotions})

  info = info.join([durations, sexes, ages, texts, emotions])

  return info, speeches

def read_ravdess_songs(sr = SR, pad = False, trunc = False):
  speeches, names, durations = list(), list(), list()

  _, dirs, _ = next(os.walk(os.path.join(DATA_PATH, "RAVDESS_songs")))
  for dirn in tqdm(dirs, ncols = 50):
      if dirn != "img":
          for fn in np.asarray(os.listdir(os.path.join(DATA_PATH, "RAVDESS_songs", dirn))):
              path = os.path.join(DATA_PATH, "RAVDESS_songs", dirn, fn)
              if not trunc:
                speech, sr = librosa.load(path, sr = sr, mono = True)
              else:
                speech, sr = librosa.load(path, sr = sr, mono = True, duration = SEGMENT_DURATION)
              indexes = librosa.effects.split(speech, top_db = 60)
              speech = np.array([num for arr in (speech[i[0]:i[1]] for i in indexes) for num in arr.tolist()])
              if pad and len(speech) < SEGMENT_DURATION*sr:
                speech = np.array([speech[i] if i < len(speech) else 0. for i in range(SEGMENT_DURATION*sr)])
              speeches.append(speech)
              names.append(fn[: -4])
              durations.append(len(speech)/sr)

  info = pd.DataFrame({"File Name": names})

  modalities, vocal_channels, emotions, emotional_intensities, statements, repetitions, sexes = list(), list(), list(), list(), list(), list(), list()

  coded_modalities =  {
    "01": "full-AV",
    "02": "video-only",
    "03": "audio-only"
  }
  coded_vocal_channels = {
    "01": "speech",
    "02": "song"
  }
  coded_emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
  }
  coded_emotional_intensities = {
    "01": "normal",
    "02": "strong"
  }
  coded_statements = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
  }
  coded_repetitions = {
    "01": "1st repetition",
    "02": "2nd repetition"
  }

  for fn in range(len(info["File Name"])):
    for i in coded_modalities:
      if info["File Name"][fn][:2] == i:
        modalities.append(coded_modalities[i])
    for i in coded_vocal_channels:
      if info["File Name"][fn][3:5] == i:
        vocal_channels.append(coded_vocal_channels[i])
    for i in coded_emotions:
      if info["File Name"][fn][6:8] == i:
        emotions.append(coded_emotions[i])
    for i in coded_emotional_intensities:
      if info["File Name"][fn][9:11] == i:
        emotional_intensities.append(coded_emotional_intensities[i])
    for i in coded_statements:
      if info["File Name"][fn][12:14] == i:
        statements.append(coded_statements[i])
    for i in coded_repetitions:
      if info["File Name"][fn][15:17] == i:
        repetitions.append(coded_repetitions[i])
    if int(info["File Name"][fn][18:20])%2 == 1:
      sexes.append("Male")
    else:
      sexes.append("Female")

  durations = pd.DataFrame({"Duration": durations})
  modalities = pd.DataFrame({"Modality": modalities})
  vocal_channels = pd.DataFrame({"Vocal_Channel": vocal_channels})
  emotions = pd.DataFrame({"Emotion": emotions})
  emotional_intensities = pd.DataFrame({"Emotional_Intensity": emotional_intensities})
  statements = pd.DataFrame({"Statement": statements})
  repetitions = pd.DataFrame({"Repetition": repetitions})
  sexes = pd.DataFrame({"Sex": sexes})

  info = info.join([durations, modalities, vocal_channels, emotions, emotional_intensities, statements, repetitions, sexes])

  return info, speeches

def read_ravdess_speeches(sr = SR, pad = False, trunc = False):
  speeches, names, durations = list(), list(), list()

  _, dirs, _ = next(os.walk(os.path.join(DATA_PATH, "RAVDESS_speeches")))
  for dirn in tqdm(dirs, ncols = 50):
    if dirn != "img":
      for fn in np.asarray(os.listdir(os.path.join(DATA_PATH, "RAVDESS_speeches", dirn))):
        path = os.path.join(DATA_PATH, "RAVDESS_speeches", dirn, fn)
        if not trunc:
          speech, sr = librosa.load(path, sr = sr, mono = True)
        else:
          speech, sr = librosa.load(path, sr = sr, mono = True, duration = SEGMENT_DURATION)
        indexes = librosa.effects.split(speech, top_db = 60)
        speech = np.array([num for arr in (speech[i[0]:i[1]] for i in indexes) for num in arr.tolist()])
        if pad and len(speech) < SEGMENT_DURATION*sr:
          speech = np.array([speech[i] if i < len(speech) else 0. for i in range(SEGMENT_DURATION*sr)])
        speeches.append(speech)
        names.append(fn[: -4])
        durations.append(len(speech)/sr)

  info = pd.DataFrame({"File Name": names})

  modalities, vocal_channels, emotions, emotional_intensities, statements, repetitions, sexes = list(), list(), list(), list(), list(), list(), list()

  coded_modalities =  {
    "01": "full-AV",
    "02": "video-only",
    "03": "audio-only"
  }
  coded_vocal_channels = {
    "01": "speech",
    "02": "song"
  }
  coded_emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
  }
  coded_emotional_intensities = {
    "01": "normal",
    "02": "strong"
  }
  coded_statements = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
  }
  coded_repetitions = {
    "01": "1st repetition",
    "02": "2nd repetition"
  }

  for fn in range(len(info["File Name"])):
    for i in coded_modalities:
      if info["File Name"][fn][:2] == i:
        modalities.append(coded_modalities[i])
    for i in coded_vocal_channels:
      if info["File Name"][fn][3:5] == i:
        vocal_channels.append(coded_vocal_channels[i])
    for i in coded_emotions:
      if info["File Name"][fn][6:8] == i:
        emotions.append(coded_emotions[i])
    for i in coded_emotional_intensities:
      if info["File Name"][fn][9:11] == i:
        emotional_intensities.append(coded_emotional_intensities[i])
    for i in coded_statements:
      if info["File Name"][fn][12:14] == i:
        statements.append(coded_statements[i])
    for i in coded_repetitions:
      if info["File Name"][fn][15:17] == i:
        repetitions.append(coded_repetitions[i])
    if int(info["File Name"][fn][18:20])%2 == 1:
      sexes.append("Male")
    else:
      sexes.append("Female")

  durations = pd.DataFrame({"Duration": durations})
  modalities = pd.DataFrame({"Modality": modalities})
  vocal_channels = pd.DataFrame({"Vocal_Channel": vocal_channels})
  emotions = pd.DataFrame({"Emotion": emotions})
  emotional_intensities = pd.DataFrame({"Emotional_Intensity": emotional_intensities})
  statements = pd.DataFrame({"Statement": statements})
  repetitions = pd.DataFrame({"Repetition": repetitions})
  sexes = pd.DataFrame({"Sex": sexes})

  info = info.join([durations, modalities, vocal_channels, emotions, emotional_intensities, statements, repetitions, sexes])

  return info, speeches

def read_soundtracks(sr = SR, pad = False, trunc = False):
  soundtracks, durations = list(), list()

  for fn in tqdm(np.asarray(os.listdir(os.path.join(DATA_PATH, "Soundtracks", "Set1"))), ncols = 50):
    path = os.path.join(DATA_PATH, "Soundtracks", "Set1", fn)
    if not trunc:
      soundtrack, sr = librosa.load(path, sr = sr, mono = True)
    else:
      soundtrack, sr = librosa.load(path, sr = sr, mono = True, duration = SEGMENT_DURATION)
    if pad and len(soundtrack) < SEGMENT_DURATION*sr:
      soundtrack = np.array([soundtrack[i] if i < len(soundtrack) else 0. for i in range(SEGMENT_DURATION*sr)])
    soundtracks.append(soundtrack)
    durations.append(len(soundtrack)/sr)

  info = pd.read_csv(os.path.join(DATA_PATH, "Soundtracks", "mean_ratings_set1.csv"))
  durations = pd.DataFrame({"Duration": durations})
  info = info.join(durations)

  return info, soundtracks

def read_datasets(database, sr = SR, pad = False, trunc = False):
  if database == "alumni":
    return read_alumni(sr)
  elif database == "emodb":
    return read_emodb(sr, pad, trunc)
  elif database == "ravdess_songs":
    return read_ravdess_songs(sr, pad, trunc)
  elif database == "ravdess_speeches":
    return read_ravdess_speeches(sr, pad, trunc)
  elif database == "soundtracks":
    return read_soundtracks(sr, pad, trunc)
  elif database == "all":
    alumni_info, alumni_speeches = read_alumni(sr)
    emodb_info, emodb_speeches = read_emodb(sr, pad, trunc)
    ravdess_songs_info, ravdess_songs = read_ravdess_songs(sr, pad, trunc)
    ravdess_speeches_info, ravdess_speeches = read_ravdess_speeches(sr, pad, trunc)
    soundtracks_info, soundtracks_songs = read_soundtracks(sr, pad, trunc)
    return [alumni_info, emodb_info, ravdess_songs_info, ravdess_speeches_info, soundtracks_info], [alumni_speeches, emodb_speeches, ravdess_songs, ravdess_speeches, soundtracks_songs]
  else:
    return 0, 0
