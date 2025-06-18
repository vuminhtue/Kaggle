import os
from scipy.io import arff
import numpy as np
from numpy.lib import recfunctions
from pathlib import Path

def read_arff_file(file_path):
    """
    Read an ARFF file and return its data and metadata.
    
    Parameters
    ----------
    file_path : str
        Path to the ARFF file
        
    Returns
    -------
    tuple
        (data, metadata) where data is a structured array and metadata contains attribute information
    """
    data, meta = arff.loadarff(file_path)
    return data, meta

def merge_arff_files(data_dir):
    """
    Merge multiple ARFF files from a directory into a single dataset.
    
    Parameters
    ----------
    data_dir : str
        Directory containing ARFF files
        
    Returns
    -------
    tuple
        (merged_data, metadata) where merged_data is a structured array containing all data
    """
    merged_data = None
    metadata = None
    
    # Get all ARFF files and sort them by year
    arff_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.arff')])
    
    for file_name in arff_files:
        year = file_name.split('.')[0]  # Extract year from filename
        file_path = os.path.join(data_dir, file_name)
        
        # Read ARFF file
        data, meta = read_arff_file(file_path)
        
        # Add year column
        year_array = np.full(len(data), year, dtype='U10')
        year_dtype = [('year', 'U10')]
        year_data = np.array([(y,) for y in year_array], dtype=year_dtype)
        
        # Merge year with original data
        if merged_data is None:
            merged_data = np.lib.recfunctions.merge_arrays([year_data, data], flatten=True)
            metadata = meta
        else:
            current_data = np.lib.recfunctions.merge_arrays([year_data, data], flatten=True)
            merged_data = np.concatenate([merged_data, current_data])
    
    return merged_data, metadata

def save_to_csv(data, output_path):
    """
    Save structured array to CSV file.
    
    Parameters
    ----------
    data : numpy.ndarray
        Structured array containing the data
    output_path : str
        Path where the CSV file should be saved
    """
    # Convert structured array to regular array with column names
    dtype_names = data.dtype.names
    data_array = np.column_stack([data[name] for name in dtype_names])
    
    # Create header
    header = ','.join(dtype_names)
    
    # Save to CSV
    np.savetxt(output_path, data_array, delimiter=',', header=header, comments='', fmt='%s')

def main():
    # Define paths
    data_dir = Path('data')
    output_path = Path('data/merged_bankruptcy_data.csv')
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge ARFF files
    print("Reading and merging ARFF files...")
    merged_data, metadata = merge_arff_files(data_dir)
    
    # Save to CSV
    print(f"Saving merged data to {output_path}...")
    save_to_csv(merged_data, output_path)
    print("Done!")

if __name__ == "__main__":
    main() 