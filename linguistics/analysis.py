from typing import Tuple
import mne
import mne_bids
import logging
import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
import toolz as Z 
from tqdm import trange

from linguistics.env import Config
from linguistics.reader.data import add_voiced_feature, add_word_frequency_feature, add_linguistic_features, clean_epochs, create_epochs, parse_annotations

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("linguistics.reader.analysis")

def run_decoding(epochs: mne.Epochs, feature: str, n_jobs: int =-1) -> pd.DataFrame:
    X = epochs.get_data() * 1e13
    y = epochs.metadata[feature].values.astype(int)
    n_splits = 5
    _, counts = np.unique(y, return_counts=True)

    if np.any(counts < n_splits):
        logger.warn(f"Warning: Cannot decode '{feature}'. Not enough samples for at least one class. Counts: {counts}")
        return pd.DataFrame()
 
    if len(set(y)) > 2: 
        y = y > np.nanmedian(y)


    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)

    n_trials, _, n_times = X.shape
    preds = np.zeros((n_trials, n_times))

    for t in trange(n_times, desc=f"Decoding {feature}"):
        predictions = cross_val_predict(
            model, X[:, :, t], y, cv=cv, method="predict_proba", n_jobs=n_jobs
        )[:, 1]
        if predictions.ndim == 1:
            print(f"Warning: predict_proba returned 1D array at timestep {t}. Check class balance.")
            continue
        preds[:, t] = predictions


    X_corr, Y_corr = y[:, None], preds
    X_corr, Y_corr = X_corr - X_corr.mean(0), Y_corr - Y_corr.mean(0)

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = (X_corr * Y_corr).sum(0) / ((X_corr**2).sum(0)**0.5 * (Y_corr**2).sum(0)**0.5)

    return pd.DataFrame(dict(score=scores, time=epochs.times))


def process_bids_file(bids_path: mne_bids.BIDSPath, phonetic_information: pd.DataFrame, n_jobs=-1) -> mne.Epochs | None:
    try:
        raw = mne_bids.read_raw_bids(bids_path)
        raw.pick_types(meg=True).load_data().filter(0.5, 30.0, n_jobs=n_jobs)
            
        meta_data = parse_annotations(raw)
        meta_data = add_voiced_feature(meta_data, phonetic_information)
        meta_data = add_word_frequency_feature(meta_data)
        meta_data = add_linguistic_features(meta_data)
 
        epochs = create_epochs((raw, meta_data))
        epochs = clean_epochs(epochs)
        
        return epochs
    except FileNotFoundError:
        return None

def to_bids_path(subject_id: str, session_id: int, task_id: int, config: Config) -> mne_bids.BIDSPath:
    return mne_bids.BIDSPath(
        subject=subject_id, session=str(session_id), task=str(task_id),
        datatype="meg", root=config.bids_root
    )

def analyze_subject(subject_id: str, config: Config, n_jobs=-1) -> Tuple[pd.DataFrame, dict]: 
    print(f"\nProcessing subject: {subject_id}")
    all_epochs = []
    for session_id in range(2):
        for task_id in range(4):
            bids_path = to_bids_path(subject_id, session_id, task_id, config)
            epochs = process_bids_file(bids_path, config.phonetic_information, n_jobs=n_jobs)
            if epochs:
                all_epochs.append(epochs)
                
    if not all_epochs:
        return pd.DataFrame(), {}
    
    subject_epochs = mne.concatenate_epochs(all_epochs)

    features_to_decode = {
        "voiced": subject_epochs["not is_word_onset"],
        "wordfreq": subject_epochs["is_word_onset"]
    }

    feature_prefixes = ['part_of_speech_', 'VerbForm_', 'Tense_', 'Number_', 'Person_', 'Mood_', 'Definite_', 'PronType_']
    morph_feature_names = [
        col for col in subject_epochs.metadata.columns
        if col.startswith(tuple(feature_prefixes))
    ]

    for feature_name in morph_feature_names:
        features_to_decode[feature_name] = subject_epochs["is_word_onset"]
    all_results = []
    for feature, epoch_subset in features_to_decode.items():
        results_df = run_decoding(epoch_subset, feature, n_jobs=n_jobs)
        results_df["contrast"] = feature
        results_df["subject"] = subject_id
        all_results.append(results_df)

    if not all_results:
        return pd.DataFrame(), {}

    final_results = pd.concat(all_results, ignore_index=True)

    return final_results, None


def analyze_all_subjects(config: Config) -> pd.DataFrame:
    all_results = []
    for subject_id in config.subjects:
        results, figs = analyze_subject(subject_id, config)
        all_results.append(results)
        # for key, fig in figs.items():
        #     fig.savefig(config.output_dir / f"{subject_id}_{key}_decoding.png")
        #     plt.close(fig)
    return pd.concat(all_results, ignore_index=True)
