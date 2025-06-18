#!/usr/bin/env python3
"""
Random Forest Feature Importance Analysis for Bankruptcy Prediction
This script trains Random Forest models and extracts feature importance rankings
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data():
    """Load and process the bankruptcy data"""
    import pandas as pd
    
    print("Loading bankruptcy datasets...")
    # Load the data
    df1 = pd.read_csv('../data/1year.arff', skiprows=69, header=None, na_values='?')
    df2 = pd.read_csv('../data/2year.arff', skiprows=69, header=None, na_values='?')
    df3 = pd.read_csv('../data/3year.arff', skiprows=69, header=None, na_values='?')
    df4 = pd.read_csv('../data/4year.arff', skiprows=69, header=None, na_values='?')
    df5 = pd.read_csv('../data/5year.arff', skiprows=69, header=None, na_values='?')
    
    # Add year labels
    df1["year"] = 1
    df2["year"] = 2
    df3["year"] = 3
    df4["year"] = 4
    df5["year"] = 5
    
    # Concatenate all datasets
    df_final = pd.concat([df1, df2, df3, df4, df5], axis=0)
    
    # Rename columns
    new_column_names = [f'X{i+1}' for i in range(64)]
    new_column_names.append('Bankrupcy')
    new_column_names.append('Year')
    df_final.columns = new_column_names
    
    print(f"Combined dataset shape: {df_final.shape}")
    print(f"Bankruptcy cases: {df_final['Bankrupcy'].sum()} ({df_final['Bankrupcy'].mean()*100:.2f}%)")
    
    return df_final

def preprocess_data(df_final):
    """Preprocess the data: handle missing values and prepare features"""
    import pandas as pd
    
    print("\nPreprocessing data...")
    
    # Check missing values
    missing_summary = df_final.isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    print(f"Features with missing values: {len(missing_features)}")
    
    if len(missing_features) > 0:
        print("Missing value summary:")
        for feature, count in missing_features.items():
            percentage = (count / len(df_final)) * 100
            print(f"  {feature}: {count} ({percentage:.2f}%)")
    
    # Remove X37 due to excessive missing values (as identified in the notebook)
    if 'X37' in df_final.columns:
        print("Removing X37 due to excessive missing values...")
        df_final = df_final.drop(columns=['X37'])
    
    # Impute missing values using KNN
    print("Imputing missing values using KNN (k=5)...")
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_final), columns=df_final.columns)
    
    # Prepare features and target
    X = df_imputed.iloc[:, :-2]  # All features except Bankrupcy and Year
    y = df_imputed['Bankrupcy'].astype(int)
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    return X, y, df_imputed

def train_random_forest_full_features(X, y):
    """Train Random Forest with all features and extract importance"""
    print("\n" + "="*60)
    print("RANDOM FOREST MODEL - ALL FEATURES")
    print("="*60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train baseline Random Forest
    print("\nTraining baseline Random Forest...")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1)
    rf_baseline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_baseline.predict(X_test)
    y_pred_proba = rf_baseline.predict_proba(X_test)[:, 1]
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Baseline Accuracy: {accuracy:.4f}")
    print(f"Baseline AUC: {auc_score:.4f}")
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf_grid = RandomForestClassifier(random_state=123, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf_grid,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation AUC: {grid_search.best_score_:.4f}")
    
    # Use the best model
    best_rf = grid_search.best_estimator_
    
    # Final predictions with best model
    y_pred_best = best_rf.predict(X_test)
    y_pred_proba_best = best_rf.predict_proba(X_test)[:, 1]
    
    accuracy_best = accuracy_score(y_test, y_pred_best)
    auc_best = roc_auc_score(y_test, y_pred_proba_best)
    
    print(f"\nOptimized Model Performance:")
    print(f"Accuracy: {accuracy_best:.4f}")
    print(f"AUC: {auc_best:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best))
    
    return best_rf, X.columns.tolist()

def extract_feature_importance(model, feature_names, top_n=20):
    """Extract and display feature importance from Random Forest model"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance
    importance_scores = model.feature_importances_
    
    # Create feature importance dataframe
    import pandas as pd
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 50)
    for i, (_, row) in enumerate(feature_importance_df.head(top_n).iterrows()):
        print(f"{i+1:2d}. {row['Feature']:6s} : {row['Importance']:.6f}")
    
    # Calculate cumulative importance
    feature_importance_df['Cumulative_Importance'] = feature_importance_df['Importance'].cumsum()
    
    # Find features that contribute to 80% and 90% of total importance
    features_80 = len(feature_importance_df[feature_importance_df['Cumulative_Importance'] <= 0.8])
    features_90 = len(feature_importance_df[feature_importance_df['Cumulative_Importance'] <= 0.9])
    
    print(f"\nFeature Importance Summary:")
    print(f"Total features: {len(feature_names)}")
    print(f"Features contributing 80% importance: {features_80}")
    print(f"Features contributing 90% importance: {features_90}")
    
    return feature_importance_df

def create_feature_importance_plots(feature_importance_df, top_n=20):
    """Create visualizations for feature importance"""
    print(f"\nCreating feature importance visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Plot 1: Top N features bar plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(top_n)
    
    bars = plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features - Random Forest')
    plt.gca().invert_yaxis()
    
    # Color bars based on importance level
    max_importance = top_features['Importance'].max()
    for i, bar in enumerate(bars):
        importance = top_features.iloc[i]['Importance']
        if importance > 0.8 * max_importance:
            bar.set_color('darkred')
        elif importance > 0.6 * max_importance:
            bar.set_color('red')
        elif importance > 0.4 * max_importance:
            bar.set_color('orange')
        else:
            bar.set_color('lightblue')
    
    plt.tight_layout()
    plt.savefig('../report/figures/rf_feature_importance_top20.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Cumulative importance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(feature_importance_df) + 1), 
             feature_importance_df['Cumulative_Importance'], 
             marker='o', markersize=2)
    plt.axhline(y=0.8, color='red', linestyle='--', label='80% threshold')
    plt.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance - Random Forest')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../report/figures/rf_cumulative_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Feature importance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(feature_importance_df['Importance'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Feature Importance')
    plt.ylabel('Number of Features')
    plt.title('Distribution of Feature Importance Scores')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../report/figures/rf_importance_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_selected_features(X, y, feature_importance_df, top_n=15):
    """Train Random Forest with only top N features and compare performance"""
    print("\n" + "="*60)
    print(f"RANDOM FOREST MODEL - TOP {top_n} FEATURES")
    print("="*60)
    
    # Select top N features
    top_features = feature_importance_df.head(top_n)['Feature'].tolist()
    X_selected = X[top_features]
    
    print(f"Selected features: {top_features}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=123, stratify=y
    )
    
    # Train Random Forest with selected features
    rf_selected = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=5,
        random_state=123, 
        n_jobs=-1
    )
    rf_selected.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_selected.predict(X_test)
    y_pred_proba = rf_selected.predict_proba(X_test)[:, 1]
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nSelected Features Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf_selected, top_features

def save_results(feature_importance_df, model_performance):
    """Save results to CSV files"""
    import pandas as pd
    import os
    
    # Create results directory
    os.makedirs('../report/results', exist_ok=True)
    
    # Save feature importance
    feature_importance_df.to_csv('../report/results/random_forest_feature_importance.csv', index=False)
    print("\nFeature importance saved to: ../report/results/random_forest_feature_importance.csv")
    
    # Save model performance summary
    performance_df = pd.DataFrame([model_performance])
    performance_df.to_csv('../report/results/random_forest_performance.csv', index=False)
    print("Model performance saved to: ../report/results/random_forest_performance.csv")

def main():
    """Main function to run the complete feature importance analysis"""
    print("="*80)
    print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
    print("BANKRUPTCY PREDICTION DATASET")
    print("="*80)
    
    # Load and process data
    df_final = load_and_process_data()
    X, y, df_imputed = preprocess_data(df_final)
    
    # Train Random Forest with all features
    best_rf_model, feature_names = train_random_forest_full_features(X, y)
    
    # Extract feature importance
    feature_importance_df = extract_feature_importance(best_rf_model, feature_names, top_n=20)
    
    # Create visualizations
    create_feature_importance_plots(feature_importance_df, top_n=20)
    
    # Analyze performance with selected features
    rf_selected, selected_features = analyze_selected_features(X, y, feature_importance_df, top_n=15)
    
    # Create comparison of top features importance
    selected_importance_df = extract_feature_importance(rf_selected, selected_features, top_n=15)
    
    # Save results
    model_performance = {
        'model_type': 'Random Forest',
        'n_features_total': len(feature_names),
        'n_features_selected': len(selected_features),
        'selected_features': ', '.join(selected_features[:10]) + '...',  # First 10 features
        'top_feature': feature_importance_df.iloc[0]['Feature'],
        'top_feature_importance': feature_importance_df.iloc[0]['Importance']
    }
    
    save_results(feature_importance_df, model_performance)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print(f"• Most important feature: {feature_importance_df.iloc[0]['Feature']}")
    print(f"• Top 3 features: {', '.join(feature_importance_df.head(3)['Feature'].tolist())}")
    print(f"• Features for 80% importance: {len(feature_importance_df[feature_importance_df['Cumulative_Importance'] <= 0.8])}")
    print(f"• Total features analyzed: {len(feature_names)}")
    
    return feature_importance_df, best_rf_model, selected_features

if __name__ == "__main__":
    feature_importance_df, model, selected_features = main() 