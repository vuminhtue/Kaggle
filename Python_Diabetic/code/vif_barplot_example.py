import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample VIF data (you can replace this with your actual vif_data)
vif_data = pd.DataFrame({
    'Feature': ['age_mid', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                'num_medications', 'number_outpatient', 'number_emergency', 
                'number_inpatient', 'number_diagnoses'],
    'VIF': [1.23, 1.45, 2.56, 1.78, 2.12, 1.56, 1.89, 2.34, 3.45]
})

# Create the plot
plt.figure(figsize=(10, 6))

# Create the barplot
ax = sns.barplot(data=vif_data, x="Feature", y="VIF", color="skyblue")

# Add title
plt.title("Variance Inflation Factor (VIF) for Numeric Features", fontsize=14)

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45, ha='right')

# Adjust layout
plt.tight_layout()

# Optionally add grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Optionally add label values on top of bars
for i, v in enumerate(vif_data['VIF']):
    ax.text(i, v + 0.1, f"{v:.2f}", ha='center')

# Save the figure
plt.savefig('vif_barplot.png')

# Display the plot (comment this if running in a script)
plt.show() 