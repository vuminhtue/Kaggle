# Student Performance Analysis and Prediction

## Project Overview
This project analyzes student performance factors and builds a predictive model using machine learning techniques. The analysis includes data preprocessing, exploratory data analysis, and Random Forest modeling to predict student exam scores.

## Repository Structure
```
├── data/
│   └── StudentPerformanceFactors.csv
├── report/
│   ├── chapters/
│   │   ├── introduction.tex
│   │   ├── eda.tex
│   │   ├── model.tex
│   │   └── conclusions.tex
│   ├── images/
│   └── main.tex
├── analyse.ipynb
└── README.md
```

## Setup Instructions
1. Create a conda environment:
```bash
conda create -n myenv python=3.11 --y
conda activate myenv
conda install numpy pandas scikit-learn seaborn matplotlib tqdm ipykernel statsmodels -y
```

2. Dataset
- The dataset is from Kaggle: [Student Success Factors and Insights](https://www.kaggle.com/datasets/anassarfraz13/student-success-factors-and-insights)
- Place the dataset in the `data` folder

## Analysis Components
1. Missing Value Analysis and Imputation
2. Exploratory Data Analysis (EDA)
   - Continuous Variable Analysis
   - Categorical Variable Analysis
3. Machine Learning Model
   - Random Forest Regression
   - Cross-validation
   - Feature Importance Analysis

## Report Generation
The LaTeX report can be compiled using:
```bash
cd report
pdflatex main.tex
```

## Key Findings
- Strong correlation between previous scores and exam performance
- Significant impact of study hours on academic performance
- Important role of teacher quality and access to resources
- Successful prediction model with high R² score

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- LaTeX