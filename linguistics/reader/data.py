
import numpy as np
import pandas as pd
import mne
from pathlib import Path

from wordfreq import zipf_frequency

def parse_annotations(raw: mne.io.Raw) -> pd.DataFrame:
    meta_list = [eval(annot.pop("description")) for annot in raw.annotations]
    return pd.DataFrame(meta_list)

def add_voiced_feature(df: pd.DataFrame, phone_information: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    phonemes = df.query('kind=="phoneme"')
    for ph, d in phonemes.groupby("phoneme"):
        ph_clean = ph.split("_")[0]
        match = phone_information.query("phoneme==@ph_clean")
        if len(match) == 1:
            df.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"
    return df

def add_word_frequency_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_word_onset"] = False
    words = df.query('kind=="word"')
    if not words.empty:
        word_onset_indices = words.index + 1
        df.loc[word_onset_indices, "is_word_onset"] = True
        wfreq = lambda x: zipf_frequency(x, "en")
        df.loc[word_onset_indices, "wordfreq"] = words.word.apply(wfreq).values
    return df

def create_epochs(data_tuple: tuple) -> mne.Epochs:
    raw, meta = data_tuple
    events = np.c_[meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))].astype(int)
    return mne.Epochs(
        raw, events, tmin=-0.200, tmax=0.6, decim=10,
        baseline=(-0.2, 0.0), metadata=meta.query('kind=="phoneme"'),
        preload=True, event_repeated="drop"
    )

def clean_epochs(epochs: mne.Epochs) -> mne.Epochs:
    epochs = epochs.copy()
    for _ in range(2):
        th = np.percentile(np.abs(epochs._data), 95)
        epochs._data[:] = np.clip(epochs._data, -th, th)
        epochs.apply_baseline()
    return epochs
