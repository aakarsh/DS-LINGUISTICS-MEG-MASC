import os

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