import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_seaborn_bar(df, x_col, y_col, title=None, figsize=(10, 6), color='skyblue', 
                     palette=None, order=None, hue=None, orient='v', 
                     rotation=45, add_labels=True):
    """
    Plot a bar chart using seaborn's barplot function.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    x_col : str
        Column name for x-axis categories
    y_col : str
        Column name for y-axis values
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size as (width, height)
    color : str, optional
        Bar color (used when palette is None and hue is None)
    palette : str or list, optional
        Color palette name or list of colors
    order : list, optional
        Order for the categorical variable (if None, sorts by y-values descending)
    hue : str, optional
        Column name for color grouping
    orient : str, optional
        'v' for vertical bars, 'h' for horizontal bars
    rotation : int, optional
        X-tick label rotation
    add_labels : bool, optional
        Whether to add value labels on top of bars
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Set figure size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create order by y value if not provided and hue is None
    if order is None and hue is None:
        if orient == 'v':
            order = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).index
        else:
            order = df.groupby(y_col)[x_col].mean().sort_values(ascending=False).index
    
    # Create the plot
    if orient == 'v':
        sns_plot = sns.barplot(
            x=x_col, y=y_col, data=df, 
            color=color, palette=palette, hue=hue, 
            order=order, ax=ax
        )
    else:
        sns_plot = sns.barplot(
            x=y_col, y=x_col, data=df, 
            color=color, palette=palette, hue=hue, 
            order=order, ax=ax
        )
    
    # Set title and labels
    if title:
        ax.set_title(title)
    
    # Set rotation for x-tick labels
    if orient == 'v':
        plt.xticks(rotation=rotation, ha='right' if rotation > 0 else 'center')
    
    # Add value labels
    if add_labels:
        if orient == 'v':
            for p in ax.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width()/2., height + (df[y_col].max() * 0.02),
                    f'{height:,.1f}', ha='center', va='bottom'
                )
        else:
            for p in ax.patches:
                width = p.get_width()
                ax.text(
                    width + (df[x_col if y_col in df else y_col].max() * 0.02), 
                    p.get_y() + p.get_height()/2.,
                    f'{width:,.1f}', ha='left', va='center'
                )
    
    plt.tight_layout()
    return fig, ax

# Example usage with the diabetic dataset
if __name__ == "__main__":
    # Set Seaborn style
    sns.set_style("whitegrid")
    
    # Load the diabetic data
    df = pd.read_csv("data/diabetic_data.csv")
    df.replace("?", np.nan, inplace=True)
    
    # Example 1: Basic vertical bar chart - Readmitted distribution
    readmitted_counts = df['readmitted'].value_counts().reset_index()
    readmitted_counts.columns = ['Readmitted_Status', 'Count']
    
    fig1, ax1 = plot_seaborn_bar(
        readmitted_counts,
        x_col='Readmitted_Status',
        y_col='Count',
        title='Distribution of Readmitted Values',
        color='royalblue',
        rotation=0
    )
    plt.savefig('seaborn_readmitted.png')
    plt.close()
    
    # Example 2: Horizontal bar chart - Missing values
    missing_values = df.isnull().sum().reset_index()
    missing_values.columns = ['Column', 'Missing_Count']
    missing_values = missing_values[missing_values['Missing_Count'] > 0]
    missing_values['Missing_Percentage'] = missing_values['Missing_Count'] / len(df) * 100
    
    fig2, ax2 = plot_seaborn_bar(
        missing_values,
        x_col='Missing_Percentage',
        y_col='Column',
        title='Percentage of Missing Values by Column',
        orient='h',  # Horizontal
        color='salmon'
    )
    plt.savefig('seaborn_missing_horizontal.png')
    plt.close()
    
    # Example 3: Bar chart with hue - Race distribution by gender
    race_gender = df.groupby(['race', 'gender']).size().reset_index(name='Count')
    
    fig3, ax3 = plot_seaborn_bar(
        race_gender,
        x_col='race',
        y_col='Count',
        title='Distribution of Race by Gender',
        hue='gender',
        palette='Set2',
        rotation=45
    )
    plt.savefig('seaborn_race_by_gender.png')
    plt.close()
    
    # Example 4: Custom color palette with order
    med_spec_counts = df['medical_specialty'].value_counts().reset_index()
    med_spec_counts.columns = ['Specialty', 'Count']
    # Only keep top 10 specialties
    top_specialties = med_spec_counts.head(10)
    
    fig4, ax4 = plot_seaborn_bar(
        top_specialties,
        x_col='Specialty',
        y_col='Count',
        title='Top 10 Medical Specialties',
        palette='viridis',  # Using a colormap
        rotation=45
    )
    plt.savefig('seaborn_top_specialties.png')
    plt.close()
    
    print("Seaborn bar charts have been generated and saved as PNG files.") 