
from linguistics.analysis import analyze_all_subjects
from linguistics.env import Config


if __name__ == "__main__":
    config = Config.load_config() 
    all_results = analyze_all_subjects(config)
    all_results.to_csv(config.output_dir / "decoding_results.csv", index=False)