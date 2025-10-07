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
subjects_dfs = []
for i in range(9, 1, -1):
    results = pd.read_csv(f"../output/0{i}_decoding_results.csv")
    fig = plotting.plot_single_subject_decoding(results, subject_id=f"0{i}", save_path=f"../output/subject_0{i}_plot.png")
    subjects_dfs.append(results)

#%%
plotting.plot_comparison_subjects(pd.concat(subjects_dfs), save_path="../output/comparison_plot.png")

#%%
all_subjects_df = pd.concat(subjects_dfs)
for contrast in all_subjects_df['contrast'].unique():
    contrast_df = all_subjects_df[all_subjects_df['contrast'] == contrast]
    fig = plotting.plot_decoding_heatmap(contrast_df, contrast_type=contrast, save_path=f"../output/{contrast}_heatmap.png")

#%%
