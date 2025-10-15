from linguistics.analysis import to_bids_path, process_epochs, concatenate_processed_epochs
from linguistics.reader.data import add_linguistic_features
from linguistics.reader.data import (
    parse_annotations
)
from linguistics.env import Config
import mne
import mne_bids
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import logging


logger = logging.getLogger("test_annotations")
logger.setLevel(logging.DEBUG)

@pytest.fixture(scope="module")
def dummy_subject_id(bids_root):
    return '01'

@pytest.fixture(scope="module")
def config():
    return Config.load_config()

@pytest.fixture(scope="module")
def bids_root(config):
    return config.bids_root

@pytest.fixture(scope="module")
def processed_epochs_for_dummy_subject(dummy_subject_id, config):
    return concatenate_processed_epochs(dummy_subject_id, config, session_range=range(1), task_range=range(1), crop_limit=None, n_jobs=-1)

def test_epochs_are_loaded_correctly(processed_epochs_for_dummy_subject):
    assert processed_epochs_for_dummy_subject is not None, "Processed epochs should not be None"
    assert isinstance(processed_epochs_for_dummy_subject, mne.BaseEpochs)
    assert not processed_epochs_for_dummy_subject.metadata.empty
    logger.info(f"Successfully loaded {len(processed_epochs_for_dummy_subject)} epochs for the dummy subject.")

def test_feature_balance_in_real_data(processed_epochs_for_dummy_subject):
    metadata = processed_epochs_for_dummy_subject.metadata

    feature_prefixes = ['part_of_speach_', 'VerbForm_', 'Tense_', 'Number_', 'Person_', 'Mood_', 'Definite_', 'PronType_']
    features_to_check = [
        col for col in metadata.columns
        if col.startswith(tuple(feature_prefixes))
    ]
    features_to_check.extend(['voiced', 'wordfreq'])

    problematic_features = []

    for feature in features_to_check:
        if feature not in metadata.columns:
            continue
        if feature == 'wordfreq':
            y_series = metadata[feature].dropna()
            y = (y_series > y_series.median()).astype(int)
            counts = y.value_counts()
        else:
            counts = metadata[feature].dropna().value_counts()

        if len(counts) < 2 or counts.min() < 5:
            problematic_features.append(f"  - '{feature}': Counts={counts.to_dict()}")
        # Log statistics which give likelihood of decoding the feature
        logger.info(
            f"Feature '{feature}': Counts={counts.to_dict()}, "
            f"Imbalance Ratio={counts.max() / counts.min() if len(counts) > 1 else 'N/A'}"
        )
    assert not problematic_features, (
        f"Found {len(problematic_features)} features that are too imbalanced for {N_SPLITS}-fold cross-validation:\n"
        + "\n".join(problematic_features)
    )
    logger.info(f"All {len(features_to_check)} checked features have sufficient samples for decoding.")

