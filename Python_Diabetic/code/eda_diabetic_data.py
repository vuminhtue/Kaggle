#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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

def main():
    print("Loading diabetic data...")
    # Load the dataset
    data_path = Path("./data/diabetic_data.csv")
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Display basic information
    print("\nBasic Information:")
    print(df.info())
    
    # Get summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # -----------------------
    # Missing Value Analysis
    # -----------------------
    print("\nAnalyzing missing values...")
    
    # Count missing values per column
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    
    if len(missing_values) > 0:
        print(f"\nColumns with missing values:\n{missing_values}")
        
        # Plot missing values
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_values.plot(kind='bar', ax=ax)
        ax.set_title('Number of Missing Values by Column')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_figure(fig, "missing_values_count.png")
        
        # Plot missing value heatmap
        plt.figure(figsize=(14, 8))
        missing_matrix = df.isnull()
        sns.heatmap(missing_matrix.sample(min(1000, len(df))), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap (Sample)')
        plt.tight_layout()
        save_figure(plt.gcf(), "missing_values_heatmap.png")
        
        # Check for special missing value indicators (like '?', 'Unknown', etc.)
        print("\nChecking for special missing value indicators...")
        special_missing = []
        
        for col in df.columns:
            # Check for '?' or 'Unknown' in string columns
            if df[col].dtype == 'object':
                val_counts = df[col].value_counts()
                for val in ['?', 'Unknown', 'NA', 'N/A', 'None']:
                    if val in val_counts.index:
                        special_missing.append((col, val, val_counts[val]))
        
        if special_missing:
            print("\nColumns with special missing value indicators:")
            for col, val, count in special_missing:
                print(f"  - {col}: '{val}' appears {count} times")
                
            # Create a dataframe for these special missing values
            special_missing_df = pd.DataFrame(special_missing, 
                                            columns=['Column', 'Indicator', 'Count'])
            special_missing_df = special_missing_df.sort_values('Count', ascending=False)
            
            # Plot special missing values
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=special_missing_df, x='Column', y='Count', hue='Indicator', ax=ax)
            ax.set_title('Special Missing Value Indicators by Column')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_figure(fig, "special_missing_values.png")
    else:
        print("No missing values found!")
    
    # -----------------------
    # Imputation Analysis
    # -----------------------
    print("\nPerforming imputation analysis...")
    
    # Select numeric columns for imputation demonstration
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        # Select a subset of numeric columns (up to 5)
        sample_cols = numeric_cols[:min(5, len(numeric_cols))]
        
        # Create a copy to avoid warnings
        df_imputed = df.copy()
        
        # Demonstration of different imputation strategies
        fig, axes = plt.subplots(len(sample_cols), 3, figsize=(18, 4*len(sample_cols)))
        
        for i, col in enumerate(sample_cols):
            # Skip if no missing values in this column
            if df[col].isnull().sum() == 0:
                # Create artificial missing values (5% of data)
                mask = np.random.rand(len(df)) < 0.05
                df_imputed[col] = df[col].copy()
                df_imputed.loc[mask, col] = np.nan
                print(f"Creating artificial missing values for {col} (5% of data)")
            
            # Original distribution
            sns.histplot(df_imputed[col].dropna(), ax=axes[i, 0], kde=True)
            axes[i, 0].set_title(f'{col} - Original Distribution')
            
            # Mean imputation
            mean_imputed = df_imputed[col].fillna(df_imputed[col].mean())
            sns.histplot(mean_imputed, ax=axes[i, 1], kde=True)
            axes[i, 1].set_title(f'{col} - Mean Imputation')
            axes[i, 1].axvline(df_imputed[col].mean(), color='red', linestyle='--')
            
            # Median imputation
            median_imputed = df_imputed[col].fillna(df_imputed[col].median())
            sns.histplot(median_imputed, ax=axes[i, 2], kde=True)
            axes[i, 2].set_title(f'{col} - Median Imputation')
            axes[i, 2].axvline(df_imputed[col].median(), color='green', linestyle='--')
        
        plt.tight_layout()
        save_figure(fig, "imputation_methods_comparison.png")
        
        # For categorical variables
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            print("\nAnalyzing categorical columns for imputation...")
            sample_cat_cols = cat_cols[:min(3, len(cat_cols))]
            
            fig, axes = plt.subplots(len(sample_cat_cols), 2, figsize=(16, 5*len(sample_cat_cols)))
            
            for i, col in enumerate(sample_cat_cols):
                # Check if multiple rows in axes
                if len(sample_cat_cols) > 1:
                    ax1, ax2 = axes[i]
                else:
                    ax1, ax2 = axes
                
                # Plot frequency of categories (top 10)
                value_counts = df[col].value_counts().nlargest(10)
                value_counts.plot(kind='bar', ax=ax1)
                ax1.set_title(f'Top Categories in {col}')
                ax1.set_ylabel('Count')
                plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
                
                # Imputation with mode demonstration
                if '?' in df[col].values or 'Unknown' in df[col].values:
                    # Replace '?' with NaN for this demonstration
                    df_cat_impute = df.copy()
                    if '?' in df[col].values:
                        df_cat_impute[col] = df_cat_impute[col].replace('?', np.nan)
                    if 'Unknown' in df[col].values:
                        df_cat_impute[col] = df_cat_impute[col].replace('Unknown', np.nan)
                    
                    # Mode imputation
                    mode_val = df_cat_impute[col].mode()[0]
                    imputed_col = df_cat_impute[col].fillna(mode_val)
                    
                    # Plot imputation result
                    imputed_counts = imputed_col.value_counts().nlargest(10)
                    imputed_counts.plot(kind='bar', ax=ax2)
                    ax2.set_title(f'{col} - Mode Imputation')
                    ax2.axhline(df_cat_impute[col].isna().sum(), color='red', 
                               linestyle='--', label=f'Missing values filled with: {mode_val}')
                    ax2.legend()
                    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                else:
                    ax2.text(0.5, 0.5, 'No standard missing values to impute', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes, fontsize=14)
            
            plt.tight_layout()
            save_figure(fig, "categorical_imputation.png")
    
    print("\nEDA completed. All figures saved to the figures directory.")

if __name__ == "__main__":
    main() 