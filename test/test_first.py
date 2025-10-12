import logging
from linguistics.env import Config 

logger = logging.getLogger("linguistics.test")
logger.setLevel(logging.DEBUG)

def test_config_loadable():
    
    config = Config.load_config()

    assert config is not None 
    assert len(config.subjects) > 0
    assert config.phonetic_information is not None
    assert config.bids_root.exists()

    logger.info(f"Subjects: {config.subjects}")
    logger.info(f"Phonetic Information: {config.phonetic_information}")
    logger.info(f"BIDS Root: {config.bids_root}")

