import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar_chart(df, x_col, y_col, title=None, figsize=(10, 6), color='skyblue', 
                  rotation=45, add_labels=True, label_offset=None):
    """
    Plot a bar chart from DataFrame columns in descending order by y values.
    
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
        Bar color
    rotation : int, optional
        X-tick label rotation
    add_labels : bool, optional
        Whether to add value labels on top of bars
    label_offset : float, optional
        Offset for labels above bars (auto-calculated if None)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Create a copy and sort by the y column in descending order
    plot_df = df[[x_col, y_col]].copy()
    plot_df = plot_df.sort_values(by=y_col, ascending=False)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(plot_df[x_col], plot_df[y_col], color=color)
    
    # Set title and labels
    if title:
        ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    # Set x-tick rotation
    plt.xticks(rotation=rotation, ha='right' if rotation > 0 else 'center')
    
    # Add value labels on top of bars
    if add_labels:
        # Auto-calculate offset if not provided
        if label_offset is None:
            y_max = plot_df[y_col].max()
            label_offset = y_max * 0.02  # 2% of max value
            
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + label_offset,
                    f'{height:,.0f}' if height >= 1 else f'{height:.2f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig, ax

# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        'Category': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'Value': [45, 90, 25, 67, 12, 58, 39]
    }
    sample_df = pd.DataFrame(data)
    
    # Plot the data
    fig, ax = plot_bar_chart(
        sample_df, 
        x_col='Category', 
        y_col='Value', 
        title='Sample Bar Chart (Descending Order)',
        color='lightgreen'
    )
    
    # Save the figure
    plt.savefig('sample_bar_chart.png')
    print("Sample bar chart saved as 'sample_bar_chart.png'")
    
    # Show the plot (comment this out if running in a script without display)
    plt.show() 