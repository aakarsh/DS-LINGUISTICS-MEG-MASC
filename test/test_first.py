from linguistics.env import load_config

def test_config_loadable():
    config = load_config()
    assert config is not None 
    assert len(config.subjects) > 0
    assert config.phonetic_information is not None
    assert config.bids_root.exists()
    

