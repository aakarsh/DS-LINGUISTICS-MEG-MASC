from linguistics.analysis import to_bids_path
from linguistics.reader.data import add_part_of_speach_feature
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
    df_with_pos = add_part_of_speach_feature(annotations_df)
    logger.info(f"DataFrame with POS feature has {len(df_with_pos)} rows")
    logger.info(f"Columns: {df_with_pos.columns.tolist()}")
    logger.info(f"Sample data:\n{df_with_pos.head(10)}")
    for row in df_with_pos.itertuples():
        if row.kind == "word":
            #logger.debug(f"Word: {row.wd}, POS: {row.part_of_speach}")
            # assert pd.notnull(row.part_of_speach), f"Word '{row.wd}' is missing part_of_speach"
            pass
        else:
            # assert pd.isnull(row.part_of_speach), f"Non-word '{row.wd}' should not have part_of_speach"
            pass

    logger.info(df_with_pos[:1000])

    assert "part_of_speach" in df_with_pos.columns
    assert not df_with_pos["part_of_speach"].isnull().all()
