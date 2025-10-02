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

## Setup:
(The following setup works for all OS: Windows, MACs, Linux. Make sure you have internet access)
- Install Visual Studio Code version > **1.104** (https://code.visualstudio.com/)
- Create Github account and sign it in VSCode. Check if the LLM is supported on the Copilot window? (https://github.com/)
- Download Anaconda Navigator for your OS (https://www.anaconda.com/products/navigator)
- Open VSCode, use terminal and create conda environment:

```
$ conda create -n myenv python=3.11 --y
$ conda activate myenv
$ conda install numpy pandas scikit-learn seaborn matplotlib tqdm ipykernel statsmodels -y 
```

## Dataset
- The dataset is from Kaggle: [Student Success Factors and Insights](https://www.kaggle.com/datasets/anassarfraz13/student-success-factors-and-insights)
- Place the dataset in the `data` folder

