# Email Spam Classification Report

This directory contains a LaTeX report documenting the analysis of email spam classification using Naive Bayes.

## Report Structure

The report consists of the following components:

- `main.tex`: Main LaTeX document
- `chapter1.tex`: Introduction to Spam and Ham Data
- `chapter2.tex`: Methodology
- `chapter3.tex`: Exploratory Data Analysis
- `chapter4.tex`: Modeling
- `chapter5.tex`: Discussion & Conclusion
- `figures/`: Directory for figures referenced in the report

## Compiling the Report

To compile the report, you need a LaTeX distribution installed on your system (such as TeX Live, MiKTeX, or MacTeX).

### Required LaTeX Packages

The report uses standard LaTeX packages:
- inputenc
- graphicx
- hyperref
- amsmath
- amssymb
- booktabs
- url

### Compilation Instructions

```bash
cd report
pdflatex main.tex
pdflatex main.tex  # Run twice for proper references
```

### Using a LaTeX Editor

You can open `main.tex` in a LaTeX editor such as TeXShop, TeXmaker, Overleaf, or Visual Studio Code with LaTeX extension, and compile from there.

## Adding Figures

The report references several figures that should be placed in the `figures/` directory:

1. `class_distribution.png`: Distribution of ham and spam emails
2. `length_distribution.png`: Distribution of email lengths by class
3. `common_words.png`: Most common words in ham and spam emails
4. `wordclouds.png`: Word clouds for ham and spam emails
5. `domain_distribution.png`: Most common email domains
6. `confusion_matrix.png`: Confusion matrix of model predictions

These figures should be generated from the analysis notebooks and placed in the figures directory before compiling the report.

## Note on Simplified Structure

This LaTeX report has been created with a simplified structure to minimize dependency issues. It uses:

- Standard verbatim environments instead of specialized code listings
- Core LaTeX packages instead of specialized formatting packages
- Simplified tables and figure references

If you have a complete LaTeX distribution installed, you can enhance the report by adding more sophisticated packages and formatting. 