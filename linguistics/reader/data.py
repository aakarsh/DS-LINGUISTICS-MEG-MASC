
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
    mark_word_onsets_df = mark_word_onsets(pd.DataFrame(meta_list))
    return mark_word_onsets_df

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
    original_columns = set(df_with_features.columns)
    df_with_features['part_of_speach'] = df_with_features.index.map(pos_map)

    morph_df = pd.DataFrame.from_dict(morph_map, orient='index')
    df_with_features = pd.concat([df_with_features, morph_df], axis=1)
    feature_cols  = ['part_of_speach'] + list(morph_df.columns)
    # Forward fill to mark the phonems rows as well.
    df_with_features[feature_cols] = df_with_features[feature_cols].ffill()
    logger.info("Applying one-hot encoding...")
    # Identify newly added morphology columns
    cols_to_encode = list(set(df_with_features.columns) - original_columns)

    for col in cols_to_encode:
        df_with_features[col] = df_with_features[col].fillna('none')

    df_with_features = pd.get_dummies(df_with_features, columns=cols_to_encode, prefix=cols_to_encode)
    logger.info(f"Encoded features: {cols_to_encode}")

    logging.debug("\n--- DEBUG: Metadata Snippet ---")
    relevant_cols = ['kind', 'word'] + [c for c in df_with_features.columns if 'Tense_' in c]
    logging.debug(df_with_features[relevant_cols].head(20).to_string())
    logging.debug("----------------------------\n")

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

def add_phonetic_features(df: pd.DataFrame, phone_information: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding phonetic features")
    df = df.copy()
    feature_cols = ['phonation', 'manner', 'place', 'frontback', 'roundness', 'centrality']
    phonemes = df.query('kind=="phoneme"')
    for ph, d in phonemes.groupby("phoneme"):
        ph_clean = ph.split("_")[0]
        match = phone_information.query("phoneme==@ph_clean")
        if len(match) == 1:
            for feature in feature_cols: 
                df.loc[d.index, feature] = match.iloc[0][feature]
    logger.info(f"Phonetic features added")
    for col in feature_cols:
        df[col] = df[col].fillna('none')
    logger.info("Applying one-hot encoding to phonetic features...")
    df = pd.get_dummies(df, columns=feature_cols, prefix=feature_cols)
    logger.info(f"Encoded features: {feature_cols}") 
    return df

def mark_word_onsets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_word"] = False
    words = df.query('kind=="word"')
    if not words.empty:
        word_onset_indices = words.index + 1
        df.loc[word_onset_indices, "is_word"] = True
    logger.info(f"Word onset feature added with {df['is_word'].sum()} word onsets")
    return df

def add_word_frequency_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    words = df.query('kind=="word"')
    if not words.empty:
        word_onset_indices = words.index + 1
        wfreq = lambda x: zipf_frequency(x, "en")
        df.loc[word_onset_indices, "wordfreq"] = words.word.apply(wfreq).values
    logger.info(f"Word frequency feature added with {df['is_word'].sum()} word onsets")
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
