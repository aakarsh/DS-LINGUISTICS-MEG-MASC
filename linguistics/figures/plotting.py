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
    
    # Formatting
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

def plot_comparison_subjects(results_df, save_path=None):
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    voiced_data = results_df[results_df['contrast'] == 'voiced']
    sns.lineplot(data=voiced_data, x='time', y='score', hue='subject', ax=axes[0])
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.7)
    axes[0].set_title('Phoneme Voicing Decoding')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Decoding Score')
    
    wordfreq_data = results_df[results_df['contrast'] == 'wordfreq']
    sns.lineplot(data=wordfreq_data, x='time', y='score', hue='subject', ax=axes[1])
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.7)
    axes[1].set_title('Word Frequency Decoding')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Decoding Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_decoding_heatmap(results_df, contrast_type='voiced', save_path=None):
    data = results_df[results_df['contrast'] == contrast_type].copy()
    heatmap_data = data.pivot(index='subject', columns='time', values='score')
    
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