# Data Science workflow 

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

### Data
- The data was downloaded from Kaggle: https://www.kaggle.com/datasets/anassarfraz13/student-success-factors-and-insights
- From VSCode, use Open Folder to navigate to your folder having data, report folder

## Let's get started with GenAI for Machine Learning 

### Steps are as follow:

```
1. Read data and check for missing values
2. If there is missing values, impute it
3. Perform EDA on categorical data and continuous data
    3.1. Heatmap for continuous data
    3.2. Boxplot+Barplot+Piechart for categorical data
    3.3. Plot with the label data
4. Train_test_split to split data to training/testing then apply 1 supervised learning approach to predict the label
5. Evaluation result using accuracy/f1-score for categorical labels and R2/RMSE for continous output

```

## Detailed prompt:
### Read the data and impute missing value
- Use pandas to read the StudentPerformanceFactors.csv file in data folder in python, calculate if there is any missing value or not and visualize using seaborn. Write simple python command and save code in analyse.ipynb 

- Impute missing value ["Teacher_Quality,Parental_Education_Level,Distance_from_Home"] using knn. If the missing values are categorical type, always convert it back to the same category and save a new csv file called Student_imputed_data.csv. Write simple python command and save code in analyse.ipynb

### EDA

- Using continuous data from Student_imputed_data.csv, calculate cross correlation between them and plot the scatter plot among all data. I also want to have heatmap for all continuos data and finally calculate VIF score (without the last column). Write simple python command and save code in analyse.ipynb 

- Using categorical data, calculate and plot the distribution all categorical data in bar chart, boxplot and piechart. 
    - For all chart type, plot them in subplot with 4 columns with corresponding title. 
    - For all chart type, color them with "Dark2" palette and be consistent between all categories, for example: positive is red, negative is blue, neutral is green, high is purple, low is grey and medium is brown, etc. 
    - For bar and boxplot chart, if the category plotting position follow from left to right: high-medium-low, yes-no, positive-neutral-negative, far-moderate-near.
    - For boxplot, use the last colum Exam_Score as the label to plot
    - Write simple python command and save code in analyse.ipynb 


### Modeling
- Use the Student_imputed_data.csv, apply One Hot Encoding to categorical data, split the training testing with ratio 70-30. Use Random Forest and cross validation approach to predict Exam_Score from all input variables and return R2 and RMSE for the prediction. Write simple python command and save code in analyse.ipynb 
- Calculate the feature importance from random forest output and visualize

### Write report
Create detailed report into report folder in Latex format using content from analyse.ipynb with the following Chapter. Save images in the same report folder
- Chapter 1: Introduction about the project, objective using the Student Performance dataset
- Chapter 2: Data and EDA (continuous and categorical features) and suggested final feature selections
- Chapter 3: Machine Learning supervised learning model using Random Forest and Cross Validation with Feature Importance
- Chapter 4: Discussion and Conclusions

```
$ pdflatex main.tex
```

### Push to github
- Use the following prompt
```
push to github
```