
import numpy as np
import pandas as pd
import mne
from pathlib import Path

from wordfreq import zipf_frequency
import logging


logger = logging.getLogger("linguistics.reader.data")
logger.setLevel(logging.DEBUG)

def parse_annotations(raw: mne.io.Raw) -> pd.DataFrame:
    meta_list = []
    for annot in raw.annotations:
        # Parse the description and merge with annotation properties
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            if k not in d:
                d[k] = v
        meta_list.append(d)
    logger.info(f"Found {len(meta_list)} annotations")
    return pd.DataFrame(meta_list)

def add_voiced_feature(df: pd.DataFrame, phone_information: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding voiced feature")
    df = df.copy()
    phonemes = df.query('kind=="phoneme"')
    for ph, d in phonemes.groupby("phoneme"):
        ph_clean = ph.split("_")[0]
        match = phone_information.query("phoneme==@ph_clean")
        if len(match) == 1:
            df.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"
    logger.info(f"Voiced feature added with {df['voiced'].sum()} voiced phonemes")
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
    logger.info(f"Word frequency feature added with {df['is_word_onset'].sum()} word onsets")
    return df

def create_epochs(data_tuple: tuple) -> mne.Epochs:
    raw, meta = data_tuple
    phoneme_meta = meta.query('kind=="phoneme"').copy()
    
    events = np.c_[
        phoneme_meta.onset * raw.info["sfreq"], 
        np.ones((len(phoneme_meta), 2))
    ].astype(int)
    
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
