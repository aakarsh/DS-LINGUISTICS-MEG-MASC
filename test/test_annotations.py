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
def cached_annotation_file():
    current_dir = Path(__file__).parent
    return current_dir / Path("caches/annotations_cache.csv")

@pytest.fixture
def annotations_df(cached_annotation_file):
    return pd.read_csv(cached_annotation_file)

def test_parse_annotations(annotations_df):
    assert isinstance(annotations_df, pd.DataFrame)
    assert not annotations_df.empty
    assert "onset" in annotations_df.columns
    assert "duration" in annotations_df.columns
    assert "word" in annotations_df.columns
    
def test_add_parts_of_speach_feature(annotations_df):
    logger.info("Testing part_of_speach feature assignment")
    df_with_pos = add_linguistic_features(annotations_df)
    logger.info(df_with_pos[:1000])
    logger.info(f"New Columns: {set(df_with_pos.columns) - set(annotations_df.columns)}")

    prefixed_cols = [col for col in df_with_pos.columns if col.startswith("part_of_speach_")]
    assert prefixed_cols, "No one-hot encoded columns found for part_of_speach"
    assert not df_with_pos[prefixed_cols].isnull().all().all(), "All one-hot encoded columns for part_of_speach are null"

def test_get_dummies(annotations_df):
    df_with_pos = add_linguistic_features(annotations_df)
    logger.info(df_with_pos.head(20))
    for col in cols_to_fill:
        dummy_cols = [c for c in df_with_pos.columns if c.startswith(f"{col}_")]
        assert dummy_cols, f"No dummy columns created for {col}"
        assert not df_with_pos[dummy_cols].isnull().all().all(), f"All dummy columns for {col} are null"
