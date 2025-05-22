import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_dataframe_bar import plot_bar_chart

# Load the diabetic data
df = pd.read_csv("data/diabetic_data.csv")
df.replace("?", np.nan, inplace=True)

# Example 1: Create a DataFrame for readmitted counts
readmitted_counts = df['readmitted'].value_counts().reset_index()
readmitted_counts.columns = ['Readmitted_Status', 'Count']

# Plot readmitted counts
fig1, ax1 = plot_bar_chart(
    readmitted_counts,
    x_col='Readmitted_Status',
    y_col='Count',
    title='Distribution of Readmitted Values',
    color='skyblue',
    rotation=0  # No rotation for this case
)
plt.savefig('readmitted_distribution_df.png')
plt.close()

# Example 2: Create a DataFrame for missing values percentage
missing_values = df.isnull().sum().reset_index()
missing_values.columns = ['Column', 'Missing_Count']
missing_values = missing_values[missing_values['Missing_Count'] > 0]
missing_values['Missing_Percentage'] = missing_values['Missing_Count'] / len(df) * 100

# Plot missing values percentage
fig2, ax2 = plot_bar_chart(
    missing_values,
    x_col='Column',
    y_col='Missing_Percentage',
    title='Percentage of Missing Values by Column',
    color='salmon',
    rotation=45
)
plt.savefig('missing_values_percentage_df.png')
plt.close()

# Example 3: Create a DataFrame for race distribution
race_counts = df['race'].value_counts().reset_index()
race_counts.columns = ['Race', 'Count']

# Plot race distribution
fig3, ax3 = plot_bar_chart(
    race_counts,
    x_col='Race',
    y_col='Count',
    title='Distribution of Race',
    color='lightgreen',
    rotation=45
)
plt.savefig('race_distribution_df.png')
plt.close()

print("Bar charts have been generated and saved as PNG files.") 