# Figure Export Template for Internet Log2 Analysis Report
# Run this after generating your plots in the notebook

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load your data (assuming df is your main dataframe)
# df = pd.read_csv('data/log2.csv', header=0)

# Figure 1: Action Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df.Action, stat='probability')
plt.title('Distribution of Network Action Types', fontsize=14, fontweight='bold')
plt.xlabel('Action Type')
plt.ylabel('Probability')
plt.tight_layout()
plt.savefig('report/figures/action_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: Port Distribution Analysis (2x2 subplots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()
port_columns = [col for col in df.columns if 'Port' in col]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']

for i, port_col in enumerate(port_columns):
    top_10_ports = df[port_col].value_counts().head(10)
    bars = axes[i].bar(range(len(top_10_ports)), top_10_ports.values, 
                       color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[i].set_title(f'{port_col}', fontsize=14, fontweight='bold', pad=20)
    axes[i].set_xlabel('Port Number', fontsize=12)
    axes[i].set_ylabel('Frequency Count', fontsize=12)
    axes[i].set_xticks(range(len(top_10_ports)))
    axes[i].set_xticklabels(top_10_ports.index, rotation=45, ha='right')
    axes[i].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout(pad=3.0)
plt.suptitle('Port Usage Distribution Analysis: Top 10 Frequencies', 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig('report/figures/port_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Correlation Heatmap
df_num = df.drop(columns=['Source Port', 'Destination Port', 'NAT Source Port', 'NAT Destination Port', 'Action'])
corr = df_num.corr(method='pearson')
plt.figure(figsize=(12, 12))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Correlation Matrix of Numerical Variables", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('report/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 4: ROC Curves Comparison (assuming you have y_SVM and y_SGD probabilities)
# You'll need to run this after training your models
"""
classes = np.unique(y)
y_bin = label_binarize(y, classes=classes)
y_test_bin = label_binarize(y_test, classes=classes)
colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# SVM ROC
ax = axes[0]
for i, class_label in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_SVM[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f'Class {class_label} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax.set_title("Multiclass ROC Curve - SVM", fontsize=14, fontweight='bold')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc='lower right')
ax.grid(True)

# SGD ROC
ax = axes[1]
for i, class_label in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_SGD[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f'Class {class_label} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
ax.set_title("Multiclass ROC Curve - SGD", fontsize=14, fontweight='bold')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc='lower right')
ax.grid(True)

plt.tight_layout()
plt.savefig('report/figures/roc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
"""

print("Figure export template created!")
print("After running your notebook analysis, execute this script to generate:")
print("1. report/figures/action_distribution.png")
print("2. report/figures/port_distribution.png") 
print("3. report/figures/correlation_heatmap.png")
print("4. report/figures/roc_comparison.png") 