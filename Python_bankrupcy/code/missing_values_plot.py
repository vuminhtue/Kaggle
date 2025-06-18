import numpy as np
import matplotlib.pyplot as plt

# Read the data files
data_files = ['../data/1year.arff', '../data/2year.arff', '../data/3year.arff', 
              '../data/4year.arff', '../data/5year.arff']

# Initialize array to store missing values
missing_counts = np.zeros(64)  # 64 features + 1 target variable

# Read and process each file
for file_path in data_files:
    with open(file_path, 'r') as f:
        # Skip header lines
        for _ in range(69):
            next(f)
        
        # Process data lines
        for line in f:
            values = line.strip().split(',')
            if len(values) == 65:  # Ensure we have all columns
                for i, val in enumerate(values):
                    if val == '?':
                        missing_counts[i] += 1

# Create feature names
feature_names = [f'X{i+1}' for i in range(64)]

# Create the plot
plt.figure(figsize=(15, 8))
plt.bar(feature_names, missing_counts)
plt.xticks(rotation=90)
plt.title('Missing Values Distribution Across Features')
plt.xlabel('Features')
plt.ylabel('Number of Missing Values')
plt.tight_layout()

# Save the plot
plt.savefig('../plots/missing_values.png', dpi=300, bbox_inches='tight')
plt.close() 