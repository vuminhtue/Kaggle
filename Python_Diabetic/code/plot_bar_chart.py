import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("data/diabetic_data.csv")
df.replace("?", np.nan, inplace=True)

# Option 1: Plot readmitted distribution
plt.figure(figsize=(10, 6))
readmitted_counts = df['readmitted'].value_counts().sort_values(ascending=False)
ax = readmitted_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Readmitted Values')
plt.xlabel('Readmitted Category')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Add count labels on top of bars
for i, count in enumerate(readmitted_counts):
    ax.text(i, count + 500, f'{count:,}', ha='center')

plt.tight_layout()
plt.savefig('readmitted_distribution.png')
plt.close()

# Option 2: Plot missing values percentage
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
missing_pct = missing_values / len(df) * 100

plt.figure(figsize=(12, 6))
ax = missing_pct.plot(kind='bar', color='salmon')
plt.title('Percentage of Missing Values by Column (Descending Order)')
plt.xlabel('Column')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45, ha='right')

# Add percentage labels on top of bars
for i, pct in enumerate(missing_pct):
    ax.text(i, pct + 1, f'{pct:.2f}%', ha='center')

plt.tight_layout()
plt.savefig('missing_values_percentage.png')
plt.close()

# Option 3: Plot categorical variable distribution (e.g., race)
plt.figure(figsize=(10, 6))
race_counts = df['race'].value_counts().sort_values(ascending=False)
ax = race_counts.plot(kind='bar', color='lightgreen')
plt.title('Distribution of Race')
plt.xlabel('Race')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Add count labels on top of bars
for i, count in enumerate(race_counts):
    ax.text(i, count + 500, f'{count:,}', ha='center')

plt.tight_layout()
plt.savefig('race_distribution.png')
print("Bar charts have been generated and saved as PNG files.") 