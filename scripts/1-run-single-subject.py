import argparse
from pathlib import Path
from linguistics import env
from linguistics.analysis import analyze_all_subjects, analyze_subject
from linguistics.env import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Run MEG decoding analysis for a single subject')
    parser.add_argument('--subject-id', type=str, required=True,
                      help='Subject ID to analyze (e.g., sub-01)')
    parser.add_argument('--max-workers', type=int, default=-1,
                      help='Maximum number of parallel workers')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = Config.load_config()
    results, figs = analyze_subject(args.subject_id, config, n_jobs=args.max_workers)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = config.output_dir / f"{args.subject_id}_decoding_results.csv"
    
    results.to_csv(output_file, index=False)