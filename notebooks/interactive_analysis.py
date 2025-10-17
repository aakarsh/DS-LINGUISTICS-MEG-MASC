#%%
print("Interactive analysis notebook for decoding results")
#%%
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path.cwd().parent  # Assumes you're running from project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import importlib
import pandas as pd

import linguistics.figures.plotting as plotting
importlib.reload(plotting)
#%%
decoding_roots = Path("../output/17-10-2025-20-34")
subjects_dfs = []
for i in range(9, 1, -1):
    if i == 3:
        continue  # Skip subject 03 due to missing data
    results = pd.read_csv(decoding_roots / f"0{i}_decoding_results.csv")
    # fig = plotting.plot_single_subject_decoding(results, subject_id=f"0{i}", save_path=decoding_roots / f"subject_0{i}_plot.png")
    subjects_dfs.append(results)

#%%
all_subjects_df = pd.concat(subjects_dfs)
for feature_name in [ 'VerbForm', 'Tense']:
    subject_ids = all_subjects_df.subject.unique()
    for selected_subject_id in subject_ids:
        subject_df = all_subjects_df[all_subjects_df['subject'] == selected_subject_id] 
        subject_df = subject_df[~subject_df['contrast'].str.endswith('_none')]
        print(subject_df['contrast'].unique())
        subjects_filtered_by_feature = subject_df[subject_df['contrast'].str.startswith(feature_name)]
        print(subjects_filtered_by_feature['contrast'].unique())
        plotting.plot_feature_comparison(subjects_filtered_by_feature, feature=feature_name, save_path="../output/feature_comparison_plot.png")
#%%
feature_name = 'Tense'
all_subjects_df = pd.concat(subjects_dfs)
selected_subject = 4
subject_df = all_subjects_df[all_subjects_df['subject'] == selected_subject]
# remove contrast VerbForm_none
subject_df = subject_df[subject_df['contrast'] != 'Tense_none']
print(subject_df['contrast'].unique())
# filter by contrast that starts with VerbForm
subjects_filtered_by_feature = subject_df[subject_df['contrast'].str.startswith(feature_name)]
print(subjects_filtered_by_feature['contrast'].unique())
plotting.plot_feature_comparison(subjects_filtered_by_feature, feature=feature_name, save_path="../output/feature_comparison_plot.png")
print("done")

#%%
plotting.plot_comparison_subjects(pd.concat(subjects_dfs), save_path="../output/comparison_plot.png")

#%%
all_subjects_df = pd.concat(subjects_dfs)
for contrast in all_subjects_df['contrast'].unique():
    contrast_df = all_subjects_df[all_subjects_df['contrast'] == contrast]
    fig = plotting.plot_decoding_heatmap(contrast_df, contrast_type=contrast, save_path=f"../output/{contrast}_heatmap.png")

#%%
