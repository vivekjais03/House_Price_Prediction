#!/usr/bin/env python3
"""
Improved House Price Prediction with Hyperparameter Tuning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def main():
    print("🚀 IMPROVED House Price Prediction Models")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    house_price_dataset = sklearn.datasets.fetch_california_housing()
    
    house_price_dataframe = pd.DataFrame(
        house_price_dataset.data, 
        columns=house_price_dataset.feature_names
    )
    house_price_dataframe['price'] = house_price_dataset.target
    
    X = house_price_dataframe.drop(['price'], axis=1)
    Y = house_price_dataframe['price']
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"✅ Data prepared! Training: {X_train.shape}, Test: {X_test.shape}")
    
    # Test multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'XGBoost (Default)': XGBRegressor(random_state=42),
        'XGBoost (Tuned)': XGBRegressor(
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
    }
    
    results = {}
    
    print("\n2. Training and evaluating multiple models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train model
        model.fit(X_train, Y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = metrics.r2_score(Y_train, train_pred)
        test_r2 = metrics.r2_score(Y_train, train_pred)
        test_r2_final = metrics.r2_score(Y_test, test_pred)
        test_rmse = np.sqrt(metrics.mean_squared_error(Y_test, test_pred))
        test_mae = metrics.mean_absolute_error(Y_test, test_pred)
        
        # Calculate overfitting
        overfitting = train_r2 - test_r2_final
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2_final,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'overfitting': overfitting
        }
        
        print(f"  ✅ {name}")
        print(f"     Test R²: {test_r2_final:.4f}")
        print(f"     Test RMSE: {test_rmse:.4f}")
        print(f"     Overfitting: {overfitting:.4f}")
    
    # Compare models
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('test_r2', ascending=False)
    
    print("\nRanked by Test R² Score:")
    print("-" * 40)
    for i, (model_name, metrics_dict) in enumerate(comparison_df.iterrows()):
        print(f"{i+1:2d}. {model_name:<20}")
        print(f"     R²: {metrics_dict['test_r2']:.4f} | RMSE: {metrics_dict['test_rmse']:.4f} | Overfitting: {metrics_dict['overfitting']:.4f}")
    
    # Best model analysis
    best_model_name = comparison_df.index[0]
    best_model = models[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Test R²: {comparison_df.iloc[0]['test_r2']:.4f}")
    print(f"   Test RMSE: {comparison_df.iloc[0]['test_rmse']:.4f}")
    
    # Feature importance for best model
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n🔍 FEATURE IMPORTANCE ({best_model_name})")
        print("-" * 40)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.iterrows()):
            print(f"{i+1:2d}. {row['Feature']:<15} {row['Importance']:.4f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    best_overfitting = comparison_df.iloc[0]['overfitting']
    if best_overfitting < 0.05:
        print("✅ Excellent generalization - model is well-balanced")
    elif best_overfitting < 0.1:
        print("👍 Good generalization - minor overfitting")
    else:
        print("⚠️  Consider regularization techniques to reduce overfitting")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Use {best_model_name} for predictions")
    print("2. Consider feature engineering")
    print("3. Collect more data if possible")
    print("4. Try ensemble methods")
    
    print("\n" + "=" * 60)
    print("🏁 Improved Analysis Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check if all required libraries are installed:")
        print("pip install numpy pandas matplotlib seaborn scikit-learn xgboost") 