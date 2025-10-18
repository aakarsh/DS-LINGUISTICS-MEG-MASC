import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

def plot_single_subject_decoding(results_df, subject_id=None, save_path=None):
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for contrast in results_df['contrast'].unique():
        data = results_df[results_df['contrast'] == contrast]
        color = 'blue' if contrast == 'voiced' else 'red'
        ax.plot(data['time'], data['score'], color=color, linewidth=2, 
                label=f'{contrast.capitalize()} Decoding')
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.7, label='Chance Level')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Decoding Score', fontsize=12)
    ax.set_title(f'MEG Decoding Results - Subject {subject_id or ""}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def plot_feature_comparison(results_df, feature, save_path=None):
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=results_df, x='time', y='score', hue='contrast', ci='sd', palette='Set2')
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.title(f'Decoding Performance by {feature.capitalize()}', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Decoding Score', fontsize=12)
    plt.legend(title=feature.capitalize(), fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return plt.gcf()

def plot_comparison_subjects(results_df, save_path=None):
    sns.set_theme(style="whitegrid")
    contrasts = results_df['contrast'].unique()
    # A grid with one row per contrast and one column
    fig, axes = plt.subplots(len(contrasts), 1, figsize=(15, 5 * len(contrasts)), sharex=False)
    for ax, contrast in zip(axes, contrasts):
        sns.lineplot(data=results_df[results_df['contrast'] == contrast], x='time', y='score', hue='subject', ax=ax)
        ax.axhline(0, color='k', linestyle='--', alpha=0.8)
        ax.set_title(f'{contrast.capitalize()} Decoding')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Decoding Score')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_decoding_heatmap(results_df, contrast_type='voiced', save_path=None):
    data = results_df[results_df['contrast'] == contrast_type].copy()
    
    # Average the scores for each subject at each time point
    subject_scores = data.groupby(['subject', 'time'])['score'].mean().reset_index()
    
    # Pivot the averaged data
    heatmap_data = subject_scores.pivot(index='subject', columns='time', values='score')
    
    #heatmap_data = data.pivot(index='label', columns='time', values='score')
    
    time_ms = heatmap_data.columns * 1000
    heatmap_data.columns = time_ms
    
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(heatmap_data, 
                cmap='YlOrRd', 
                cbar_kws={'label': 'Decoding Score'},
                xticklabels=20, 
                yticklabels=True)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Subjects')
    plt.title(f'{contrast_type.capitalize()} Decoding Performance')
  
    zero_col_idx = np.argmin(np.abs(heatmap_data.columns.values))
    plt.axvline(x=zero_col_idx, color='black', linewidth=2)  
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()