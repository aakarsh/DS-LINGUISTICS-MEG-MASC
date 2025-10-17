from linguistics.analysis import to_bids_path, process_epochs, concatenate_processed_epochs, run_decoding
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
    logger.info("Testing if processed epochs are loaded correctly...")
    assert processed_epochs_for_dummy_subject is not None, "Processed epochs should not be None"
    assert isinstance(processed_epochs_for_dummy_subject, mne.BaseEpochs)
    assert not processed_epochs_for_dummy_subject.metadata.empty
    logger.info(f"Successfully loaded {len(processed_epochs_for_dummy_subject)} epochs for the dummy subject.")

def test_feature_balance_in_real_data(processed_epochs_for_dummy_subject):
    logger.info("Testing feature balance in real data...")
    metadata = processed_epochs_for_dummy_subject.metadata
    logger.info(f"Metadata columns: {metadata.columns.tolist()}")
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
    logger.info(f"Checked {len(features_to_check)} features for balance, Found {len(problematic_features)} problematic features.")
    logger.info(f"Problematic features: {problematic_features}")
    # Assert majority of features are balanced enough for decoding
    assert len(problematic_features) < len(features_to_check) / 2, (
        f"Too many features are imbalanced for 5-fold cross-validation:\n"
        + "\n".join(problematic_features)
    )
    logger.info(f"All {len(features_to_check)} checked features have sufficient samples for decoding.")


def test_run_decoding_voiced(processed_epochs_for_dummy_subject):
    logger.info("Testing decoding for 'voiced' feature...")
    epochs = processed_epochs_for_dummy_subject
    feature = 'voiced'
    epoch_subset = epochs['not is_word']
    if 'voiced' not in epochs.metadata.columns:
        pytest.skip("'voiced' feature not found in metadata.")
    results_df = run_decoding(epoch_subset, feature, n_jobs=-1)
    # should have scores
    score_column = results_df['score']
    assert not score_column.isnull().all(), "Decoding scores should not be all NaN"
    results_df['contrast'] = feature
    logger.debug(f"Decoding results for '{feature}':\n{results_df}")
    assert not results_df.empty, "Decoding results should not be empty"


def test_run_decoding_wordfreq(processed_epochs_for_dummy_subject):
    logger.info("Testing decoding for 'voiced' feature...")
    epochs = processed_epochs_for_dummy_subject
    feature = 'wordfreq'
    epoch_subset = epochs['is_word']
    if 'wordfreq' not in epochs.metadata.columns:
        pytest.skip("'wordfreq' feature not found in metadata.")
    results_df = run_decoding(epoch_subset, feature, n_jobs=-1)
    # should have scores
    score_column = results_df['score']
    assert not score_column.isnull().all(), "Decoding scores should not be all NaN"
    results_df['contrast'] = feature
    logger.debug(f"Decoding results for '{feature}':\n{results_df}")
    assert not results_df.empty, "Decoding results should not be empty"

def test_run_decoding_parts_of_speech(processed_epochs_for_dummy_subject):
    logger.info("Testing decoding for 'parts_of_speech' feature...")
    epochs = processed_epochs_for_dummy_subject
    feature = 'part_of_speach_NOUN'
    epoch_subset = epochs['is_word']
    if feature not in epochs.metadata.columns:
        pytest.skip(f"'{feature}' feature not found in metadata.")
    results_df = run_decoding(epoch_subset, feature, n_jobs=-1)
    # should have scores
    score_column = results_df['score']
    assert not score_column.isnull().all(), "Decoding scores should not be all NaN"
    results_df['contrast'] = feature
    logger.debug(f"Decoding results for '{feature}':\n{results_df}")
    assert not results_df.empty, "Decoding results should not be empty"
