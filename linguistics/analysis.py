from typing import Tuple
from typing import List
import mne
import mne_bids
import logging
import numpy as np
import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
import toolz as Z 
from tqdm import trange

from linguistics.env import Config
from linguistics.reader.data import add_voiced_feature,add_phonetic_features, add_word_frequency_feature, add_linguistic_features, clean_epochs, create_epochs, parse_annotations

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("linguistics.reader.analysis")
logger.setLevel(logging.DEBUG)

all_word_morph_feature_prefixes = ['part_of_speach', 'VerbForm', 'Tense', 'Number', 'Person', 'Mood', 'Definite', 'PronType']
all_phonetic_feature_prefixes = ['phonation', 'manner', 'place', 'frontback', 'roundness', 'centrality']

ALL_FEATURE_PREFIXES = all_word_morph_feature_prefixes + all_phonetic_feature_prefixes

def is_word_feature_prefix(feature: str) -> bool:
    return any(feature.startswith(prefix) for prefix in all_word_morph_feature_prefixes)

def is_phonetic_feature_prefix(feature: str) -> bool:
    return any(feature.startswith(prefix) for prefix in all_phonetic_feature_prefixes)

def run_decoding(epochs: mne.Epochs, feature: str,n_splits: int=5, n_jobs: int =-1) -> pd.DataFrame:
    stats = {
        "feature": feature,
        "n_classes": 0,
        "class_counts": [],
        "mean_diff_rms": np.nan,
        "snr_estimate": np.nan
    }

    X_full = epochs.get_data() * 1e13
    y_series = epochs.metadata[feature]

    # Add a tiny amount of noise to prevent zero variance issues
    X_full += 1e-12 * np.random.randn(*X_full.shape)

    valid_trials = ~y_series.isna()
    meta_valid = epochs.metadata[valid_trials].copy()

    if not np.any(valid_trials):
        logger.warning(f"Warning: No valid (non-NaN) data for feature '{feature}'. Skipping.")
        return pd.DataFrame()

    X = X_full[valid_trials]
    y = y_series[valid_trials].values.astype(int)
    #    If so, binarize it by splitting it at the median.
    if pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) > 2:
        y = (y > np.nanmedian(y)).astype(int)
    else:
        y = y.astype(int)

    unique_labels, counts = np.unique(y, return_counts=True)
    stats["n_classes"] = len(unique_labels)
    stats["class_counts"] = counts

    if len(counts) < 2 or np.any(counts < n_splits):
        logger.warn(f"Warning: Cannot decode '{feature}'. Not enough samples for at least one class. Counts: {counts}")
        return pd.DataFrame()
 
    if len(set(y)) > 2: 
        y = y > np.nanmedian(y)

    mean_class_0 = X[y == 0].mean(axis=0)
    mean_class_1 = X[y == 1].mean(axis=0)


    diff = mean_class_0 - mean_class_1
    stats["mean_diff_rms"] = np.sqrt(np.mean(diff**2))


    var_class_0 = X[y == 0].var(axis=0)
    var_class_1 = X[y == 1].var(axis=0)


    mean_diff_sq = np.mean(diff**2)
    mean_within_var = np.mean(var_class_0) + np.mean(var_class_1)

    if mean_within_var > 0:
        stats["snr_estimate"] = mean_diff_sq / mean_within_var

    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)

    logger.info(f"Statistics for feature '{feature}': {stats}")

    # --- START DEBUGGING SNIPPET ---
    logger.info(f"--- Debugging Folds for feature: '{feature}' ---")
    unique_labels, counts = np.unique(y, return_counts=True)
    logger.info(f"Overall class distribution: {dict(zip(unique_labels, counts))}")

    cv_debug = cv
    fold_is_problematic = False
    for i, (train_index, _) in enumerate(cv_debug.split(X, y)):
        y_train = y[train_index]
        train_labels, train_counts = np.unique(y_train, return_counts=True)
        logger.info(f"  Fold {i} training set distribution: {dict(zip(train_labels, train_counts))}")
        if len(train_labels) < 2:
            logger.error(f"  !! Problem in Fold {i}: Only one class present in training data.")
            fold_is_problematic = True

    if fold_is_problematic:
        logger.error("At least one fold is invalid. Stopping before full decoding.")
        return pd.DataFrame() # Stop execution for this feature
    logger.info("--- Fold debugging complete. All folds are valid. ---")
    # --- END DEBUGGING SNIPPET ---

    n_trials, _, n_times = X.shape
    preds = np.zeros((n_trials, n_times))
    '''
    for t in trange(n_times, desc=f"Decoding {feature}"):
        predictions = cross_val_predict(
            model, X[:, :, t], y, cv=cv, method="predict_proba", n_jobs=n_jobs
        )[:, 1]
        # This always happens not sure why.
        if predictions.ndim == 1:
            logger.warn(f"Warning: predict_proba returned 1D array at timestep {t}. Check class balance.")
            continue
        preds[:, t] = predictions
    '''
    # Initialize a predictions array with NaNs to store probabilities for the positive class
    preds = np.full((n_trials, n_times), np.nan)

    # --- MANUAL CROSS-VALIDATION LOOP ---
    logger.info("--- Starting Manual Cross-Validation Loop ---")
    for t in trange(n_times, desc=f"Decoding {feature}"):
        Xt = X[:, :, t]


        # This will store predictions for the current time point across all folds
        preds_t = np.full(n_trials, np.nan)

        for fold_idx, (train_index, test_index) in enumerate(cv.split(Xt, y)):
            X_train, X_test = Xt[train_index], Xt[test_index]
            y_train, y_test = y[train_index], y[test_index]

            stds = np.std(X_train, axis=0)
            # Check if any of the standard deviations are zero
            if np.any(stds == 0):
                logger.warning(
                    f"  -> FOLD {fold_idx}, TIMESTEP {t}: "
                    f"Zero variance detected in {np.sum(stds == 0)} out of {len(stds)} channels. "
                    "StandardScaler will fail."
                )


            train_labels, train_counts = np.unique(y_train, return_counts=True)
            if len(train_labels) < 2:
                logger.error(f"  !! FOLD {fold_idx}, TIMESTEP {t}: Only one class in training data! Labels: {train_labels}. Skipping fold.")
                continue

            try:
                cloned_model = clone(model)
                cloned_model.fit(X_train, y_train)

                learned_classes = cloned_model.named_steps['lineardiscriminantanalysis'].classes_
                if len(learned_classes) < 2:
                    logger.warning(f"  -> FOLD {fold_idx}, TIMESTEP {t}: Model only learned one class: {learned_classes}.")
                    probas = cloned_model.predict_proba(X_test) # will be 1D
                    if learned_classes[0] == 0:
                        preds_t[test_index] = 0.0 # Prob of class 1 is 0
                    else:
                        preds_t[test_index] = 1.0 # Prob of class 1 is 1
                else:
                    probas = cloned_model.predict_proba(X_test)[:, 1]
                    preds_t[test_index] = probas

            except Exception as e:
                logger.error(f"  !! ERROR in FOLD {fold_idx}, TIMESTEP {t}: {e}", exc_info=True)
                break

        preds[:, t] = preds_t

    logger.info("--- Manual Cross-Validation Loop Finished ---")


    # --- END MANUAL CROSS-VALIDATION LOOP ---
    # --- SCORING ---
    out = list()
    for label, m in meta_valid.groupby("label"):
        logger.info(f"Scoring label: {label} with {len(m.index)} trials.")
        group_indices = meta_valid.index.get_indexer_for(m.index)

        Rs = correlate(y[group_indices, None], preds[group_indices])
        for t, r in zip(epochs.times, Rs):
            out.append(dict(score=r, time=t, label=label, n=len(m.index)))

    '''
        X_corr, Y_corr = y[:, None], preds
        X_corr, Y_corr = X_corr - X_corr.mean(0), Y_corr - Y_corr.mean(0)

        with np.errstate(divide='ignore', invalid='ignore'):
            scores = (X_corr * Y_corr).sum(0) / ((X_corr**2).sum(0)**0.5 * (Y_corr**2).sum(0)**0.5)

        return pd.DataFrame(dict(score=scores, time=epochs.times))
    '''
    return pd.DataFrame(out)

def correlate(X, Y):
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    X, Y = X - X.mean(0), Y - Y.mean(0)

    SXY = (X * Y).sum(0)

    SX2 = (X**2).sum(0) ** 0.5
    SY2 = (Y**2).sum(0) ** 0.5

    return SXY / (SX2 * SY2)

def process_bids_file(bids_path: mne_bids.BIDSPath, phonetic_information: pd.DataFrame, n_jobs=-1, crop_limit=None) -> mne.Epochs | None:
    try:
        raw = mne_bids.read_raw_bids(bids_path)
        if crop_limit:
            raw.crop(tmax=crop_limit)
        raw.pick_types(meg=True).load_data().filter(0.5, 30.0, n_jobs=n_jobs)
            
        meta_data = parse_annotations(raw)
        logger.info(f"Parsed {len(meta_data)} annotations from raw data.")


        meta_data = add_voiced_feature(meta_data, phonetic_information)
        meta_data = add_phonetic_features(meta_data, phonetic_information)
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

def process_epochs(subject_id: str, config: Config, session_range = range(2), task_range=range(4),  crop_limit=None, n_jobs=-1) -> List[mne.Epochs]:
    all_epochs = []
    for session_id in session_range:
        for task_id in task_range:
            bids_path = to_bids_path(subject_id, session_id, task_id, config)
            epochs = process_bids_file(bids_path, config.phonetic_information, n_jobs=n_jobs, crop_limit = crop_limit)
            epochs.metadata["task"] = task_id
            epochs.metadata["half"] = np.round(
                np.linspace(0, 1.0, len(epochs))
            ).astype(int)
            epochs.metadata["session"] = session_id
            if epochs:
                all_epochs.append(epochs)
    return all_epochs

def epoch_labeling(m: pd.DataFrame) -> pd.Series:
    label = (
            "t"
            + m.task.astype(str)
            + "_s"
            + m.session.astype(str)
            + "_h"
            + m.half.astype(str)
        )
    return label

def concatenate_processed_epochs(subject_id: str, config:  Config, session_range = range(2), task_range=range(4), crop_limit=None, n_jobs=-1):
    logger.info(f"  -> Processing epochs for subject {subject_id}...")
    all_epochs = process_epochs(subject_id, config, session_range = session_range, task_range=task_range, n_jobs=n_jobs)
    epochs = mne.concatenate_epochs(all_epochs)
    epochs.metadata["label"] = epoch_labeling(epochs.metadata)
    logger.info(f"  -> Concatenated {len(all_epochs)} epoch sets. Total epochs: {len(epochs)}")
    return epochs

def analyze_subject(subject_id: str, config: Config, n_jobs=-1, feature_prefixes=ALL_FEATURE_PREFIXES) -> Tuple[pd.DataFrame, dict]:
    print(f"\nProcessing subject: {subject_id} Number of features to decode: {len(feature_prefixes) + 2}")

    cache_dir = config.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{subject_id}-epo.fif"

    if cache_path.exists():
        logger.info(f"  -> Found cache, loading preprocessed epochs from: {cache_path}")
        subject_epochs = mne.read_epochs(cache_path, preload=True)
    else:
        subject_epochs = concatenate_processed_epochs(subject_id, config, n_jobs=n_jobs)
        logger.info(f"  -> Saving preprocessed epochs to cache: {cache_path}")
        subject_epochs.save(cache_path, overwrite=True)

    features_to_decode = { }
    
    for feature in feature_prefixes:
        if is_word_feature_prefix(feature):
            features_to_decode[feature] = subject_epochs["is_word"]
        elif is_phonetic_feature_prefix(feature):
            features_to_decode[feature] = subject_epochs["not is_word"] 
            
    logging.debug("\n--- Verifying Feature Diversity ---")
    n_splits = 5
    decodable_features = []
    undecodable_features = {}

    for feature, epoch_subset in features_to_decode.items():
        if feature in epoch_subset.metadata.columns:
            counts = epoch_subset.metadata[feature].dropna().value_counts()
            if len(counts) >= 2 and counts.min() >= n_splits:
                decodable_features.append(feature)
            else:
                undecodable_features[feature] = counts.to_dict()

    print(f"Found {len(decodable_features)} decodable features.")
    if undecodable_features:
        print("\nSkipping features with insufficient samples:")
        for feature, counts in undecodable_features.items():
            print(f"  - '{feature}': Counts={counts}")
    logger.debug("-------------------------------------\n")

    all_results = []
    for feature, epoch_subset in features_to_decode.items():
        results_df = run_decoding(epoch_subset, feature, n_jobs=n_jobs)
        results_df["contrast"] = feature
        results_df["subject"] = subject_id
        all_results.append(results_df)

    if not all_results:
        return pd.DataFrame(), {}

    final_results = pd.concat(all_results, ignore_index=True)
    debug_path = config.output_dir / f"{subject_id}_metadata_for_debug.csv"
    print(f"--- DEBUG: Speichere Metadaten zur Überprüfung in {debug_path} ---")
    subject_epochs.metadata.to_csv(debug_path)
    # ------------------------------------

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
