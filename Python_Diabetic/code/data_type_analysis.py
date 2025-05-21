#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set plot style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    # Use a more modern alternative if the old style is not available
    try:
        plt.style.use('seaborn-whitegrid')  # Try this first
    except:
        plt.style.use('default')  # Fall back to default if needed

sns.set_palette('viridis')

# Create figures directory if it doesn't exist
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

# Create subdirectories for categorical and numerical plots
cat_dir = figures_dir / "categorical"
num_dir = figures_dir / "numerical"
cat_dir.mkdir(exist_ok=True)
num_dir.mkdir(exist_ok=True)

def save_figure(fig, filename, subdir=None):
    """Save a figure to the figures directory"""
    if subdir:
        save_path = figures_dir / subdir / filename
    else:
        save_path = figures_dir / filename
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def identify_variable_types(df, max_categories=20):
    """
    Identify categorical and numerical variables in a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    max_categories : int
        Maximum number of unique values for a column to be considered categorical
        
    Returns:
    --------
    dict
        Dictionary with keys 'categorical' and 'numerical', each containing a list of column names
    """
    categorical = []
    numerical = []
    
    for col in df.columns:
        # Skip ID columns (usually not informative for analysis)
        if 'id' in col.lower() or 'identifier' in col.lower():
            continue
            
        # Check if column data type is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # If numeric but has few unique values, might be categorical
            if df[col].nunique() <= max_categories:
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            # For non-numeric types, check number of unique values
            if df[col].nunique() <= max_categories:
                categorical.append(col)
            else:
                # Text columns with many unique values (might be free text)
                pass
    
    return {'categorical': categorical, 'numerical': numerical}

def convert_age_to_numeric(df):
    """Convert age ranges to numeric values"""
    if 'age' in df.columns:
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Convert age ranges to numeric values
        def parse_age(age_str):
            if pd.isna(age_str) or age_str == '?':
                return np.nan
            
            try:
                if age_str == '>90':
                    return 95
                lower, upper = map(int, age_str.split('-'))
                return (lower + upper) / 2
            except Exception as e:
                print(f"Error converting age '{age_str}': {e}")
                return np.nan
        
        # Apply the conversion and create a new column
        df['age_numeric'] = df['age'].apply(parse_age)
    
    return df

def plot_categorical_variable(df, col, top_n=10):
    """Create a bar plot for a categorical variable"""
    # Get value counts, limiting to top N categories if there are many
    val_counts = df[col].value_counts().nlargest(top_n)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    bars = ax.bar(
        val_counts.index, 
        val_counts.values,
        color=plt.cm.viridis(np.linspace(0, 0.8, len(val_counts)))
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + (val_counts.max() * 0.01),
            f'{height:,}',
            ha='center', va='bottom', fontsize=9
        )
    
    # Add percentage labels as a second line
    total = val_counts.sum()
    for i, (rect, val) in enumerate(zip(bars, val_counts)):
        percentage = (val / total) * 100
        ax.text(
            rect.get_x() + rect.get_width()/2.,
            rect.get_height() + (val_counts.max() * 0.03),
            f'({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=8, alpha=0.7
        )
    
    # Set labels and title
    ax.set_title(f'Distribution of {col.replace("_", " ").title()}', fontsize=12)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_xlabel(col.replace("_", " ").title(), fontsize=10)
    
    # Rotate x-labels if there are many or they're long
    if val_counts.index.dtype != 'int64' or len(val_counts) > 5 or max([len(str(x)) for x in val_counts.index]) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Add a grid for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, f"{col}_distribution.png", "categorical")
    
    return fig

def plot_categorical_paired_analysis(df, col, target_col='readmitted', top_n=10):
    """Create a stacked percentage bar chart showing relationship between a categorical variable and a target"""
    if target_col not in df.columns or col == target_col:
        # Skip self-comparisons and invalid target columns
        return None
    
    # Check if the column has any data
    if df[col].nunique() <= 1:
        print(f"  - Skipping {col} - not enough unique values for analysis")
        return None
    
    # Get the top categories
    top_categories = df[col].value_counts().nlargest(top_n).index
    
    # Filter dataframe to include only top categories
    filtered_df = df[df[col].isin(top_categories)]
    
    try:
        # Create a cross-tabulation
        cross_tab = pd.crosstab(
            filtered_df[col], 
            filtered_df[target_col],
            normalize='index'
        ) * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create stacked percentage bar chart
        cross_tab.plot(
            kind='bar', 
            stacked=True,
            ax=ax,
            colormap='viridis'
        )
        
        # Set labels and title
        ax.set_title(f'Relationship between {col.replace("_", " ").title()} and {target_col.replace("_", " ").title()}', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_xlabel(col.replace("_", " ").title(), fontsize=10)
        
        # Add legend
        ax.legend(title=target_col.replace("_", " ").title())
        
        # Rotate x-labels
        plt.xticks(rotation=45, ha='right')
        
        # Add a grid for easier reading
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        save_figure(fig, f"{col}_vs_{target_col}.png", "categorical")
        
        return fig
    except TypeError as e:
        print(f"  - Skipping {col} vs {target_col} - {str(e)}")
        return None
    except Exception as e:
        print(f"  - Error creating paired analysis for {col} vs {target_col}: {str(e)}")
        return None

def plot_numerical_distributions(df, numerical_vars, bins=30, max_cols=4, max_vars=12):
    """
    Create a grid of histograms for numerical variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    numerical_vars : list
        List of numerical variable names
    bins : int
        Number of bins for histograms
    max_cols : int
        Maximum number of columns in the grid
    max_vars : int
        Maximum number of variables to plot
    """
    # Limit the number of variables to plot
    if len(numerical_vars) > max_vars:
        print(f"Too many numerical variables. Plotting only the first {max_vars}.")
        numerical_vars = numerical_vars[:max_vars]
    
    # Calculate grid dimensions
    n_vars = len(numerical_vars)
    n_cols = min(n_vars, max_cols)
    n_rows = int(np.ceil(n_vars / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Flatten axes array for easy iteration
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each variable
    for i, var in enumerate(numerical_vars):
        if i < len(axes):  # Safety check
            # Create histogram with KDE
            sns.histplot(df[var].dropna(), bins=bins, kde=True, ax=axes[i])
            
            # Add variable statistics
            stats_text = (
                f"Mean: {df[var].mean():.2f}\n"
                f"Median: {df[var].median():.2f}\n"
                f"Std: {df[var].std():.2f}"
            )
            axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
            
            # Set title and labels
            axes[i].set_title(var.replace("_", " ").title())
            axes[i].set_xlabel("")
            
            # Add grid
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, "numerical_distributions.png", "numerical")
    
    return fig

def plot_boxplots(df, numerical_vars, max_cols=4, max_vars=12):
    """Create a grid of boxplots for numerical variables"""
    # Limit the number of variables to plot
    if len(numerical_vars) > max_vars:
        print(f"Too many numerical variables. Plotting only the first {max_vars}.")
        numerical_vars = numerical_vars[:max_vars]
    
    # Calculate grid dimensions
    n_vars = len(numerical_vars)
    n_cols = min(n_vars, max_cols)
    n_rows = int(np.ceil(n_vars / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Flatten axes array for easy iteration
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each variable
    for i, var in enumerate(numerical_vars):
        if i < len(axes):  # Safety check
            # Create boxplot
            sns.boxplot(y=df[var].dropna(), ax=axes[i])
            
            # Set title
            axes[i].set_title(var.replace("_", " ").title())
            
            # Add grid
            axes[i].grid(axis='x', linestyle='--', alpha=0.7)
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, "numerical_boxplots.png", "numerical")
    
    return fig

def plot_correlation_heatmap(df, numerical_vars, max_vars=15):
    """Create a correlation heatmap for numerical variables"""
    # Limit the number of variables if there are too many
    if len(numerical_vars) > max_vars:
        print(f"Too many numerical variables for correlation heatmap. Using top {max_vars} with highest variance.")
        # Select variables with highest variance
        vars_std = df[numerical_vars].std().sort_values(ascending=False)
        numerical_vars = vars_std.index[:max_vars].tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_vars].corr()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap='viridis',
        vmax=1.0, vmin=-1.0, center=0,
        annot=True, fmt='.2f',
        square=True, linewidths=.5
    )
    
    # Set title
    plt.title('Correlation Matrix of Numerical Variables', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_figure(plt.gcf(), "correlation_heatmap.png", "numerical")
    
    return plt.gcf()

def plot_target_relationships(df, numerical_vars, target_col='readmitted', max_vars=6):
    """Create boxplots showing relationship between numerical variables and a target"""
    if target_col not in df.columns:
        return None
    
    # Limit the number of variables
    if len(numerical_vars) > max_vars:
        # Select variables with highest correlation to target (if target is numeric)
        # or highest variance otherwise
        if pd.api.types.is_numeric_dtype(df[target_col]):
            correlations = df[numerical_vars + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
            numerical_vars = [col for col in correlations.index if col != target_col][:max_vars]
        else:
            vars_std = df[numerical_vars].std().sort_values(ascending=False)
            numerical_vars = vars_std.index[:max_vars].tolist()
    
    # Calculate grid dimensions
    n_vars = len(numerical_vars)
    n_cols = min(n_vars, 3)
    n_rows = int(np.ceil(n_vars / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Flatten axes array for easy iteration
    if n_vars > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each variable
    for i, var in enumerate(numerical_vars):
        if i < len(axes):  # Safety check
            # Create boxplot
            sns.boxplot(x=df[target_col], y=df[var], ax=axes[i])
            
            # Set title and labels
            axes[i].set_title(f'{var.replace("_", " ").title()} by {target_col.replace("_", " ").title()}')
            axes[i].set_xlabel(target_col.replace("_", " ").title())
            axes[i].set_ylabel(var.replace("_", " ").title())
            
            # Add grid
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x-labels if needed
            if df[target_col].nunique() > 3:
                plt.sca(axes[i])
                plt.xticks(rotation=45, ha='right')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_figure(fig, f"numerical_by_{target_col}.png", "numerical")
    
    return fig

def analyze_and_visualize_dataframe(df, target_col='readmitted'):
    """
    Analyze a dataframe, split variables into categorical and numerical,
    and create appropriate visualizations for each type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Name of the target column for paired analyses
    """
    print("Beginning data analysis...")
    
    # Convert age to numeric if present
    df = convert_age_to_numeric(df)
    
    # Identify variable types
    var_types = identify_variable_types(df)
    categorical_vars = var_types['categorical']
    numerical_vars = var_types['numerical']
    
    print(f"Identified {len(categorical_vars)} categorical variables and {len(numerical_vars)} numerical variables.")
    
    # Print the lists
    print("\nCategorical variables:")
    for var in categorical_vars:
        print(f"  - {var} ({df[var].nunique()} unique values)")
    
    print("\nNumerical variables:")
    for var in numerical_vars:
        print(f"  - {var} (min: {df[var].min()}, max: {df[var].max()}, mean: {df[var].mean():.2f})")
    
    # Analyze categorical variables
    print("\nAnalyzing categorical variables...")
    
    for var in categorical_vars:
        print(f"  - Creating distribution plot for {var}")
        plot_categorical_variable(df, var)
        
        if target_col in df.columns:
            print(f"  - Creating paired analysis for {var} vs {target_col}")
            plot_categorical_paired_analysis(df, var, target_col)
    
    # Analyze numerical variables
    print("\nAnalyzing numerical variables...")
    
    # Plot distributions
    print("  - Creating distribution plots for numerical variables")
    plot_numerical_distributions(df, numerical_vars)
    
    # Plot boxplots
    print("  - Creating boxplots for numerical variables")
    plot_boxplots(df, numerical_vars)
    
    # Plot correlation heatmap
    print("  - Creating correlation heatmap for numerical variables")
    plot_correlation_heatmap(df, numerical_vars)
    
    # Plot relationships with target
    if target_col in df.columns:
        print(f"  - Creating plots showing relationships between numerical variables and {target_col}")
        plot_target_relationships(df, numerical_vars, target_col)
    
    print("\nAnalysis complete! All visualizations have been saved to the figures directory.")
    
    return {
        'categorical': categorical_vars,
        'numerical': numerical_vars
    }

def main():
    print("Loading diabetic data...")
    # Load the dataset
    data_path = Path("data/diabetic_data.csv")
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Analyze the dataframe
    var_info = analyze_and_visualize_dataframe(df, 'readmitted')
    
    print("\nType breakdown summary:")
    print(f"Categorical variables: {len(var_info['categorical'])}")
    print(f"Numerical variables: {len(var_info['numerical'])}")
    print("\nVisualization complete! Check the 'figures' directory for all plots.")

if __name__ == "__main__":
    main() 