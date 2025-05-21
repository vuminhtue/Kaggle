#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Create figures directory if it doesn't exist
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

def save_figure(fig, filename):
    """Save a figure to the figures directory"""
    fig.savefig(figures_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_top_diagnoses(df, n_top=15):
    """
    Create 3 subplots showing the top diagnoses for diag_1, diag_2, and diag_3
    with sorted value counts and values displayed on top of each bar.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the diabetic data
    n_top : int
        Number of top diagnoses to display in each subplot
    """
    # Create a figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Define the diagnosis columns
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    
    # Loop through each diagnosis column and create a subplot
    for i, col in enumerate(diag_cols):
        # Get value counts and sort in descending order
        value_counts = df[col].value_counts().nlargest(n_top)
        
        # Create bar plot
        bars = axes[i].bar(
            x=range(len(value_counts)), 
            height=value_counts.values,
            color=plt.cm.viridis(np.linspace(0, 0.8, len(value_counts)))
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width()/2.,
                height + (value_counts.max() * 0.01),  # Small offset above bar
                f'{height:,}',  # Format with commas for thousands
                ha='center', va='bottom', rotation=0,
                fontsize=9, fontweight='normal'
            )
        
        # Set x-axis labels with ICD9 codes
        axes[i].set_xticks(range(len(value_counts)))
        axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # Set titles and labels
        axes[i].set_title(f'Top {n_top} Diagnoses: {col.replace("_", " ").title()}', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].set_xlabel('ICD9 Code', fontsize=10)
        
        # Add a grid for easier reading
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add percentage labels as a second line
        percentage = (value_counts / df[col].count()) * 100
        for j, (rect, pct) in enumerate(zip(bars, percentage)):
            axes[i].text(
                rect.get_x() + rect.get_width()/2.,
                rect.get_height() + (value_counts.max() * 0.03),  # Position above count
                f'({pct:.1f}%)',
                ha='center', va='bottom', fontsize=8, alpha=0.7
            )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, "top_diagnoses.png")
    
    # Show the figure
    plt.show()
    
    return fig

def plot_diagnosis_heatmap(df, n_top=15):
    """
    Create a heatmap showing the relationships between top diagnoses across the three diagnosis columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the diabetic data
    n_top : int
        Number of top diagnoses to include
    """
    # Get top diagnoses from each column
    top_diag1 = df['diag_1'].value_counts().nlargest(n_top).index.tolist()
    top_diag2 = df['diag_2'].value_counts().nlargest(n_top).index.tolist()
    top_diag3 = df['diag_3'].value_counts().nlargest(n_top).index.tolist()
    
    # Create a cross-tabulation of top diagnoses from diag_1 and diag_2
    cross_tab_12 = pd.crosstab(
        df['diag_1'][df['diag_1'].isin(top_diag1)], 
        df['diag_2'][df['diag_2'].isin(top_diag2)],
        normalize='index'
    ) * 100
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(cross_tab_12, cmap='viridis', annot=True, fmt='.1f', linewidths=0.5)
    plt.title('Co-occurrence of Top Diagnoses (%) - Primary vs. Secondary Diagnoses')
    plt.tight_layout()
    
    # Save the figure
    save_figure(plt.gcf(), "diagnosis_co_occurrence.png")
    
    # Show the figure
    plt.show()

def main():
    print("Loading diabetic data...")
    # Load the dataset
    data_path = Path("data/diabetic_data.csv")
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Plot top diagnoses
    plot_top_diagnoses(df, n_top=15)
    
    # Plot diagnosis heatmap
    plot_diagnosis_heatmap(df, n_top=10)
    
    print("\nDiagnosis visualizations completed. All figures saved to the figures directory.")

if __name__ == "__main__":
    main() 