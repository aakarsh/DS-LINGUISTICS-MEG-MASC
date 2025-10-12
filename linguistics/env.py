from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

default_paths = {
    "PROJECT_ROOT": "/projects/DS-LINGUISTICS-MEG-MASC",
    "DATA_DIR": "/projects/data/meg-masc",
    "LOCAL_DATA_DIR": "/projects/DS-LINGUISTICS-MEG-MASC/data",
    "OUTPUT_DIR": "/projects/DS-LINGUISTICS-MEG-MASC/output",
}

def project_root_directory():
    return os.getenv("PROJECT_ROOT", default_paths["PROJECT_ROOT"])

def data_directory():
    return os.getenv("DATA_DIR", default_paths["DATA_DIR"])

def output_directory():
    return os.getenv("OUTPUT_DIR", default_paths["OUTPUT_DIR"])

def local_data_directory():
    return os.getenv("LOCAL_DATA_DIR", default_paths["LOCAL_DATA_DIR"])

def phoneme_inventory():
    return os.path.join(local_data_directory(), "phoneme_info.csv")

@dataclass
class Config:
    bids_root: Path
    output_dir: Path 
    phonetic_information: pd.DataFrame
    subjects: list
    
    
    @staticmethod
    def load_config(bids_root = data_directory(), phoneme_path = phoneme_inventory()) -> "Config":
        bids_root = Path(bids_root)
        ph_info_path = Path(phoneme_path)
        phonetic_information = pd.read_csv(ph_info_path)
        
        subjects_df = pd.read_csv(bids_root / "participants.tsv", sep="\t")
        subjects = subjects_df.participant_id.apply(lambda x: x.split("-")[1]).tolist()

        return Config(bids_root=bids_root, phonetic_information=phonetic_information, subjects=subjects, output_dir=Path(output_directory()))
