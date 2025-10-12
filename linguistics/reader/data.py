
from matplotlib.pylab import annotations
import numpy as np
import pandas as pd
import mne
from pathlib import Path

from wordfreq import zipf_frequency
import logging
import spacy
import toolz as Z
logger = logging.getLogger("linguistics.reader.data")
logger.setLevel(logging.DEBUG)

#spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

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

class AnnotationsHelper:
    
    @staticmethod
    def parse_annotations(raw: mne.io.Raw) -> pd.DataFrame:
        return parse_annotations(raw)
    
    @staticmethod
    def add_voiced_feature(df: pd.DataFrame, phone_information: pd.DataFrame) -> pd.DataFrame:
        return add_voiced_feature(df, phone_information)
    
    @staticmethod
    def add_word_frequency_feature(df: pd.DataFrame) -> pd.DataFrame:
        return add_word_frequency_feature(df)
    
    @staticmethod
    def add_part_of_speach_feature(df: pd.DataFrame) -> pd.DataFrame:
        return add_part_of_speach_feature(df)

def add_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding part-of-speach feature")
    df = df.copy()
    words_df = df[df['kind'] == 'word'].dropna(subset=['word'])

    # Full text to character mapping
    full_text = ""
    char_to_original_index_map = {}
    for index, row in words_df.iterrows():
        start_char = len(full_text)
        full_text += str(row['word']) + " "
        char_to_original_index_map[start_char] = index

    pos_map = Z.pipe(
        nlp(full_text.strip()),
        # Create a dictionary of {original_index: pos_tag}
        lambda doc: {
            char_to_original_index_map.get(token.idx): token.pos_
            for token in doc
            if token.idx in char_to_original_index_map
        }
    )

    morph_map = Z.pipe(
        nlp(full_text.strip()),
        # Create a dictionary of {original_index: morph}
        lambda doc: {
            char_to_original_index_map.get(token.idx): token.morph.to_dict()
            for token in doc
            if token.idx in char_to_original_index_map
        }
    )

    df_with_features = df.copy()
    df_with_features['part_of_speach'] = df_with_features.index.map(pos_map)
    morph_df = pd.DataFrame.from_dict(morph_map, orient='index')
    df_with_features = pd.concat([df_with_features, morph_df], axis=1)

    return df_with_features

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

    phoneme_events = np.c_[
        phoneme_meta.onset * raw.info["sfreq"],
        np.ones((len(phoneme_meta), 2))
    ].astype(int)
    
    return mne.Epochs(
        raw, phoneme_events, tmin=-0.200, tmax=0.6, decim=10,
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
