from linguistics.analysis import to_bids_path
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
def config():
    return Config.load_config()

@pytest.fixture
def bids_root(config):
    return config.bids_root

@pytest.fixture
def processed_bids_file(bids_root):
    current_dir = Path(__file__).parent
    return current_dir / Path("caches/processed_bids_cache.fif")

def test_first_decoding_test():
    pass
