import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

# Create figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# Load data
df = pd.read_csv("../data/diabetic_data.csv")
df.replace("?", np.nan, inplace=True)

# 1. Readmission distribution
plt.figure(figsize=(8, 6))
readmit_counts = df['readmitted'].value_counts().reset_index()
readmit_counts.columns = ['readmitted', 'count']
sns.barplot(data=readmit_counts, x='readmitted', y='count', hue='readmitted', palette='Set2')
plt.title('Readmission Count')
plt.xlabel('Readmitted')
plt.ylabel('Number of Patients')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../figures/readmission_distribution.pdf', bbox_inches='tight')
plt.close()

# 2. Missing values percentage
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
missing_pct = missing_values / len(df) * 100

plt.figure(figsize=(10, 6))
missing_pct.plot(kind='bar', color='coral')
plt.title('Percentage of Missing Values by Variable')
plt.ylabel('Percentage Missing')
plt.xlabel('Variables')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../figures/missing_values_percentage.pdf', bbox_inches='tight')
plt.close()

# Drop irrelevant columns for further analysis
df = df.drop(["encounter_id", "patient_nbr"], axis=1)

# Age conversion
age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}
df['age_mid'] = df['age'].map(age_map)
df = df.drop(["age"], axis=1)

# 3. Prepare categorical variables
cat_cols = df.select_dtypes("object").columns
df_cat = pd.concat([df[cat_cols], df[["admission_type_id", "discharge_disposition_id", "admission_source_id"]]], axis=1)
df_num = df.drop(df_cat.columns, axis=1)
df_cat = df_cat.drop(["readmitted"], axis=1)

# 4. Categorical variables grid (simplified for demonstration)
# Select a few important categorical variables
selected_cat_vars = ["race", "gender", "medical_specialty", "insulin", "diabetesMed", "admission_type_id"]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(selected_cat_vars):
    # Get counts by category and readmission status
    counts = df.groupby([col, 'readmitted']).size().unstack(fill_value=0)
    
    # Calculate percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    
    # Plot
    percentages.plot(kind='bar', stacked=True, ax=axes[i], colormap='Set2')
    axes[i].set_title(f'{col}')
    axes[i].set_ylabel('Percentage (%)')
    axes[i].set_xlabel('')
    axes[i].legend(title='Readmitted')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../figures/categorical_variables_grid.pdf', bbox_inches='tight')
plt.close()

# 5. Correlation matrix
corr = df_num.corr(method="pearson")
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('../figures/correlation_matrix.pdf', bbox_inches='tight')
plt.close()

# 6. VIF analysis
vif_data = pd.DataFrame()
vif_data['Feature'] = df_num.columns
vif_data['VIF'] = [variance_inflation_factor(df_num.values, i) for i in range(df_num.shape[1])]
vif_data = vif_data.sort_values(by="VIF", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=vif_data, x="Feature", y="VIF")
plt.title("VIF for Numerical Features")
plt.xticks(rotation=45, ha="right")
plt.ylabel("VIF")
plt.tight_layout()
plt.savefig('../figures/vif_barplot.pdf', bbox_inches='tight')
plt.close()

# 7. Numerical distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
palette = sns.color_palette("Set2", 9)

for i, column in enumerate(df_num.columns):
    sns.histplot(df_num[column], bins=10, color=palette[i], ax=axes[i])
    axes[i].set_title(f'Distribution of {column}')

plt.tight_layout()
plt.savefig('../figures/numerical_distributions.pdf', bbox_inches='tight')
plt.close()

# 8. Numerical boxplots by readmission status
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i, col in enumerate(df_num.columns):
    sns.boxplot(data=df, x='readmitted', y=col, ax=axes[i], palette="Set2")
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('../figures/numerical_boxplots.pdf', bbox_inches='tight')
plt.close()

# 9. Create a simplified model for confusion matrix and feature importance
# One-hot encode selected categorical variables
df_cat_selected = df[["race", "gender", "medical_specialty", "insulin", "diabetesMed", "admission_type_id"]]
df_cat_encoded = pd.get_dummies(df_cat_selected, drop_first=True)

# Combine with numerical features
X = pd.concat([df_num, df_cat_encoded], axis=1)
y = df['readmitted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a simple model (for demonstration)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 10. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NO', '>30', '<30'],
            yticklabels=['NO', '>30', '<30'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../figures/confusion_matrix.pdf', bbox_inches='tight')
plt.close()

# 11. Feature importance
# Get feature importance for each class
classes = model.classes_
n_classes = len(classes)
importance = pd.DataFrame(model.coef_, columns=X.columns, index=classes)

# Plot top 15 features for each class
fig, axes = plt.subplots(1, n_classes, figsize=(18, 10))

for i, cls in enumerate(classes):
    # Sort and select top 15 features for this class
    sorted_data = importance.loc[cls].abs().sort_values(ascending=False).head(15)
    
    # Bar plot
    sns.barplot(x=sorted_data.values, y=sorted_data.index, ax=axes[i], palette=['#5A9BD4'])
    axes[i].set_title(f'Feature Importance: {cls}')
    axes[i].set_xlabel('Absolute Coefficient')
    axes[i].set_ylabel('Feature' if i == 0 else '')

plt.tight_layout()
plt.savefig('../figures/feature_importance.pdf', bbox_inches='tight')
plt.close()

print("All figures have been saved to the '../figures' directory.") 