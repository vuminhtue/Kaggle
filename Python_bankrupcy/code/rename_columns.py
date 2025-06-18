import numpy as np

# Read the CSV file
data = np.genfromtxt('data/merged_bankruptcy_data.csv', delimiter=',', skip_header=1)

# Get the number of columns
n_columns = data.shape[1]

# Create new column names with X prefix
new_column_names = [f'X{i+1}' for i in range(n_columns)]

# Save the data with new column names
header = ','.join(new_column_names)
np.savetxt('data/merged_bankruptcy_data_renamed.csv', data, delimiter=',', 
           header=header, comments='', fmt='%s')

print(f"Columns renamed to: {new_column_names}") 