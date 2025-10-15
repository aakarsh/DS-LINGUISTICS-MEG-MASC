from linguistics.analysis import to_bids_path, process_epochs
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

@pytest.fixture
def dummy_subject_id(bids_root):
    return '01'

@pytest.fixture
def config():
    return Config.load_config()

@pytest.fixture
def bids_root(config):
    return config.bids_root

@pytest.fixture
def processed_bids_file(bids_root):
    current_dir = Path(__file__).parent
    return current_dir / Path("caches/processed_bids_cache.fif")

@pytest.fixture
def processed_epochs_for_dummy_subject(dummy_subject_id):
    return process_epochs(dummy_subject_id, Config.load_config(), session_range=range(1), task_range=range(1), n_jobs=-1)

def test_first_decoding_test(processed_epochs_for_dummy_subject):
    assert processed_epochs_for_dummy_subject is not None, "Processed epochs should not be None"
