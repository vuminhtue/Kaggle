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

def convert_age_to_numeric(age_str):
    """
    Convert age range string to numeric middle value.
    
    Parameters:
    -----------
    age_str : str
        Age range string (e.g., '0-10', '10-20', etc.)
        
    Returns:
    --------
    float
        Middle value of the age range
    """
    if pd.isna(age_str) or age_str == '?':
        return np.nan
    
    try:
        # Handle special case for age > 90
        if age_str == '>90':
            return 95  # Assuming 95 as middle point for >90
        
        # Split the range and calculate middle value
        lower, upper = map(int, age_str.split('-'))
        return (lower + upper) / 2
    except Exception as e:
        print(f"Error converting age '{age_str}': {e}")
        return np.nan

def age_analysis(df):
    """
    Analyze the age distribution after conversion to numeric values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the diabetic data with converted age column
    """
    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original categorical age distribution
    age_counts = df['age'].value_counts().sort_index()
    axes[0].bar(age_counts.index, age_counts.values, color='skyblue')
    axes[0].set_title('Original Age Group Distribution')
    axes[0].set_xlabel('Age Group')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Numeric age distribution
    sns.histplot(df['age_numeric'], bins=20, kde=True, ax=axes[1])
    axes[1].set_title('Numeric Age Distribution (Mid-Range Values)')
    axes[1].set_xlabel('Age (Numeric - Mid-Range)')
    axes[1].set_ylabel('Count')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add statistics annotation
    stats_text = (
        f"Mean: {df['age_numeric'].mean():.1f}\n"
        f"Median: {df['age_numeric'].median():.1f}\n"
        f"Std Dev: {df['age_numeric'].std():.1f}\n"
        f"Min: {df['age_numeric'].min():.1f}\n"
        f"Max: {df['age_numeric'].max():.1f}"
    )
    axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, "age_distribution.png")
    
    return fig

def main():
    print("Loading diabetic data...")
    # Load the dataset
    data_path = Path("data/diabetic_data.csv")
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Check if 'age' column exists
    if 'age' not in df.columns:
        print("Error: 'age' column not found in the dataset")
        return
    
    # Display unique values in the age column
    print("\nUnique age values before conversion:")
    print(df['age'].unique())
    
    # Convert age to numeric
    print("\nConverting age ranges to numeric middle values...")
    df['age_numeric'] = df['age'].apply(convert_age_to_numeric)
    
    # Display the first few rows to verify the conversion
    print("\nFirst few rows after conversion:")
    print(df[['age', 'age_numeric']].head())
    
    # Calculate statistics on the numeric age column
    print("\nAge statistics after conversion:")
    print(df['age_numeric'].describe())
    
    # Analyze and visualize age distribution
    print("\nCreating age distribution visualization...")
    age_analysis(df)
    
    # Save the converted dataset if needed
    # df.to_csv("data/diabetic_data_processed.csv", index=False)
    
    print("\nAge conversion and analysis completed. Visualization saved to the figures directory.")

if __name__ == "__main__":
    main() 