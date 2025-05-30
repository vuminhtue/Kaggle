#!/usr/bin/env python3
# read_ham.py - Read ham emails and organize them into structured data

import os
import numpy as np
from email.parser import Parser

def read_email_files(directory):
    """
    Read all files in the specified directory and extract their contents.
    
    Args:
        directory (str): Path to the directory containing email files
        
    Returns:
        tuple: (filenames, contents, labels) as numpy arrays
    """
    filenames = []
    contents = []
    labels = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Parse each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
                
            filenames.append(filename)
            contents.append(content)
            labels.append(1)  # Label as 1 for ham emails
        except Exception as e:
            print(f"Error reading file {filename}: {str(e)}")
    
    # Convert to numpy arrays
    return np.array(filenames), np.array(contents), np.array(labels)

def save_to_csv(filenames, contents, labels, output_file):
    """
    Save the email data to a CSV file.
    
    Args:
        filenames (np.array): Array of filenames
        contents (np.array): Array of email contents
        labels (np.array): Array of labels
        output_file (str): Path to output CSV file
    """
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['filename', 'content', 'label'])
        
        for i in range(len(filenames)):
            # Replace newlines in content to avoid breaking CSV format
            clean_content = contents[i].replace('\n', ' ').replace('\r', '')
            writer.writerow([filenames[i], clean_content, labels[i]])
    
    print(f"Data saved to {output_file}")

def main():
    # Path to easy_ham directory
    easy_ham_dir = os.path.join('..', 'data', 'easy_ham')
    
    # Read email files
    filenames, contents, labels = read_email_files(easy_ham_dir)
    
    print(f"Read {len(filenames)} emails from {easy_ham_dir}")
    
    # Display first few entries
    for i in range(min(3, len(filenames))):
        print(f"\nFile: {filenames[i]}")
        print(f"Label: {labels[i]}")
        print(f"Content (first 100 chars): {contents[i][:100]}...")
    
    # Save data to CSV file (optional)
    output_file = os.path.join('..', 'data', 'ham_data.csv')
    save_to_csv(filenames, contents, labels, output_file)
    
    # Return the data as a tuple
    return filenames, contents, labels

if __name__ == "__main__":
    main() 