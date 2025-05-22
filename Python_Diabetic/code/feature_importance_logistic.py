import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_top_features_from_logistic(model, feature_names, top_n=5):
    """
    Extract top n important features from a trained logistic regression model.
    
    Parameters:
    -----------
    model : trained LogisticRegression model
        The trained logistic regression model
    feature_names : list or array
        Names of the features used in the model
    top_n : int, optional
        Number of top features to return (default: 5)
        
    Returns:
    --------
    top_features_df : pandas DataFrame
        DataFrame with feature names and their importance scores
    """
    # For binary classification
    if len(model.classes_) == 2:
        # Get coefficients (absolute values for importance)
        importance = np.abs(model.coef_[0])
        
        # Create a DataFrame of features and their importance scores
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
    
    # For multi-class classification
    else:
        # Get mean absolute coefficient across all classes as importance
        importance = np.mean(np.abs(model.coef_), axis=0)
        
        # Create a DataFrame of features and their importance scores
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
    
    # Sort by importance (descending) and get top n
    top_features = feature_importance.sort_values('Importance', ascending=False).head(top_n)
    
    return top_features

# Example usage with the diabetic dataset
if __name__ == "__main__":
    # Load the data (replace with your actual data loading)
    # For demonstration, I'll create a simplified example
    
    # Load your diabetic data
    df = pd.read_csv("data/diabetic_data.csv")
    df.replace("?", np.nan, inplace=True)
    
    # Prepare data for logistic regression
    # Select some numeric columns for demonstration
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                    'num_medications', 'number_outpatient', 'number_emergency', 
                    'number_inpatient', 'number_diagnoses', 'age']
    
    # Convert age to numeric if it's categorical
    if df['age'].dtype == 'object':
        # Use age mapping similar to your notebook
        age_map = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        df['age'] = df['age'].map(age_map)
    
    # Select only rows with no missing values in these columns
    X = df[numeric_cols].dropna()
    
    # Get the corresponding target values
    y = df.loc[X.index, 'readmitted']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    
    # Get top 5 important features
    top_features = get_top_features_from_logistic(model, X.columns, top_n=5)
    
    print("Top 5 Important Features:")
    print(top_features)
    
    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=top_features, x='Feature', y='Importance', color='skyblue')
    
    # Add title and labels
    plt.title('Top 5 Important Features in Logistic Regression Model', fontsize=14)
    plt.xlabel('Features')
    plt.ylabel('Importance (|Coefficient|)')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(top_features['Importance']):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('top_features_logistic.png')
    plt.show() 