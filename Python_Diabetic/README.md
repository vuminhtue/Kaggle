# Diabetic Data Analysis and Modeling

This repository contains a comprehensive analysis of diabetic patient data, focusing on hospital readmission patterns. The analysis includes exploratory data analysis, data preprocessing, and predictive modeling using logistic regression with K-fold cross-validation.

## Project Structure

- **code/**: Contains Jupyter notebooks and Python scripts for analysis
  - `eda_diabetic_data.ipynb`: Main analysis notebook
  - `save_figures.py`: Script to generate figures for the LaTeX report
  - Additional utility scripts for specific analyses

- **data/**: Contains the dataset files
  - `diabetic_data.csv`: Main dataset (101,766 observations × 50 variables)
  - `IDs_mapping.csv`: Mapping of ID values to descriptions

- **figures/**: Contains generated plots and visualizations for the report

- **report/**: Contains the LaTeX report files
  - `main.tex`: Main LaTeX document
  - `chapters/`: Individual chapter files
  - `Makefile`: For compiling the report

## Dataset Overview

The dataset contains hospital records of diabetic patients from 1999-2008 at 130 U.S. hospitals. It includes patient demographics, admission details, diagnoses, lab tests, medications, and readmission outcomes. The target variable "readmitted" has three categories: "NO" (not readmitted), ">30" (readmitted after 30 days), and "<30" (readmitted within 30 days).

## Report Content

The LaTeX report consists of four chapters:

1. **Introduction**: Background on diabetes, readmission issues, dataset description, and research objectives
2. **Exploratory Data Analysis**: Data overview, missing value analysis, categorical and numerical variable analysis
3. **Modeling with Logistic Regression**: Model selection, training with K-fold cross-validation, evaluation, and feature importance analysis
4. **Conclusions**: Summary of findings, clinical implications, limitations, and future research directions

## Generating the Report

To compile the LaTeX report:

1. Ensure LaTeX is installed on your system
2. Generate figures by running the Python script:
   ```
   cd code
   python save_figures.py
   ```
3. Compile the LaTeX document:
   ```
   cd report
   make
   ```
4. The compiled PDF will be available as `report/main.pdf`

To clean up auxiliary files:
```
cd report
make clean
```

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- statsmodels
- LaTeX distribution (for report compilation) 