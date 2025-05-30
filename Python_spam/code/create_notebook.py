#!/usr/bin/env python3
# create_notebook.py - Create a Jupyter notebook for reading ham emails

import json

# Define the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Ham Email Data Processing\n", "\n", "This notebook reads email files from the `easy_ham` directory and organizes them into structured data using NumPy arrays."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["import os\n", "import numpy as np\n", "from email.parser import Parser"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def read_email_files(directory):\n",
                "    \"\"\"\n",
                "    Read all files in the specified directory and extract their contents.\n",
                "    \n",
                "    Args:\n",
                "        directory (str): Path to the directory containing email files\n",
                "        \n",
                "    Returns:\n",
                "        tuple: (filenames, contents, labels) as numpy arrays\n",
                "    \"\"\"\n",
                "    filenames = []\n",
                "    contents = []\n",
                "    labels = []\n",
                "    \n",
                "    # Check if directory exists\n",
                "    if not os.path.exists(directory):\n",
                "        raise FileNotFoundError(f\"Directory not found: {directory}\")\n",
                "    \n",
                "    # Parse each file in the directory\n",
                "    for filename in os.listdir(directory):\n",
                "        file_path = os.path.join(directory, filename)\n",
                "        \n",
                "        # Skip directories\n",
                "        if os.path.isdir(file_path):\n",
                "            continue\n",
                "            \n",
                "        try:\n",
                "            with open(file_path, 'r', encoding='latin-1') as f:\n",
                "                content = f.read()\n",
                "                \n",
                "            filenames.append(filename)\n",
                "            contents.append(content)\n",
                "            labels.append(1)  # Label as 1 for ham emails\n",
                "        except Exception as e:\n",
                "            print(f\"Error reading file {filename}: {str(e)}\")\n",
                "    \n",
                "    # Convert to numpy arrays\n",
                "    return np.array(filenames), np.array(contents), np.array(labels)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Read Ham Emails"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Path to easy_ham directory\n",
                "easy_ham_dir = os.path.join('..', 'data', 'easy_ham')\n",
                "\n",
                "# Read email files\n",
                "filenames, contents, labels = read_email_files(easy_ham_dir)\n",
                "\n",
                "print(f\"Read {len(filenames)} emails from {easy_ham_dir}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Examine Sample Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display first few entries\n",
                "for i in range(min(3, len(filenames))):\n",
                "    print(f\"\\nFile: {filenames[i]}\")\n",
                "    print(f\"Label: {labels[i]}\")\n",
                "    print(f\"Content (first 100 chars): {contents[i][:100]}...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Data Structure Analysis"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Examine the structure of our data\n",
                "print(f\"Number of emails: {len(filenames)}\")\n",
                "print(f\"Data types: {filenames.dtype}, {contents.dtype}, {labels.dtype}\")\n",
                "\n",
                "# Create a structured view of the data\n",
                "data_structure = np.column_stack([\n",
                "    np.arange(len(filenames)),  # Index\n",
                "    filenames,                 # Filename\n",
                "    labels                     # Label\n",
                "])\n",
                "\n",
                "# Display a sample in a structured form\n",
                "print(\"\\nSample data structure (index, filename, label):\")\n",
                "for i in range(min(5, len(data_structure))):\n",
                "    print(f\"{int(float(data_structure[i][0])):>5}  {data_structure[i][1]:<30}  {int(float(data_structure[i][2]))}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write the notebook to file
with open('read_ham.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully: read_ham.ipynb") 