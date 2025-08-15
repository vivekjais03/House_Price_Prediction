#!/usr/bin/env python3
"""
House Price Prediction using XGBoost
California Housing Dataset Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

def main():
    print("🏠 House Price Prediction using XGBoost")
    print("=" * 50)
    
    # 1. Load Dataset
    print("\n1. Loading California Housing Dataset...")
    house_price_dataset = sklearn.datasets.fetch_california_housing()
    print(f"✅ Dataset loaded! Shape: {house_price_dataset.data.shape}")
    
    # 2. Create DataFrame
    print("\n2. Creating DataFrame...")
    house_price_dataframe = pd.DataFrame(
        house_price_dataset.data, 
        columns=house_price_dataset.feature_names
    )
    house_price_dataframe['price'] = house_price_dataset.target
    print(f"✅ DataFrame created! Shape: {house_price_dataframe.shape}")
    
    # 3. Data Exploration
    print("\n3. Exploring Data...")
    print(f"Features: {list(house_price_dataframe.columns[:-1])}")
    print(f"Target: price")
    print(f"Missing values: {house_price_dataframe.isnull().sum().sum()}")
    
    # 4. Data Splitting
    print("\n4. Splitting Data...")
    X = house_price_dataframe.drop(['price'], axis=1)
    Y = house_price_dataframe['price']
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print(f"✅ Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 5. Model Training
    print("\n5. Training XGBoost Model...")
    model = XGBRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, Y_train)
    print("✅ Model training completed!")
    
    # 6. Model Evaluation
    print("\n6. Evaluating Model...")
    
    # Training predictions
    train_pred = model.predict(X_train)
    train_r2 = metrics.r2_score(Y_train, train_pred)
    train_mae = metrics.mean_absolute_error(Y_train, train_pred)
    train_rmse = np.sqrt(metrics.mean_squared_error(Y_train, train_pred))
    
    # Test predictions
    test_pred = model.predict(X_test)
    test_r2 = metrics.r2_score(Y_test, test_pred)
    test_mae = metrics.mean_absolute_error(Y_test, test_pred)
    test_rmse = np.sqrt(metrics.mean_squared_error(Y_test, test_pred))
    
    # 7. Results
    print("\n" + "=" * 50)
    print("📊 MODEL PERFORMANCE RESULTS")
    print("=" * 50)
    
    print(f"\nTraining Data:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    
    print(f"\nTest Data:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    
    # 8. Feature Importance
    print(f"\n🔍 FEATURE IMPORTANCE")
    print("-" * 30)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for i, (_, row) in enumerate(feature_importance.iterrows()):
        print(f"{i+1:2d}. {row['Feature']:<15} {row['Importance']:.4f}")
    
    # 9. Overfitting Analysis
    print(f"\n⚠️  OVERFITTING ANALYSIS")
    print("-" * 30)
    overfitting_r2 = train_r2 - test_r2
    overfitting_rmse = test_rmse - train_rmse
    
    print(f"R² difference: {overfitting_r2:.4f}")
    print(f"RMSE difference: {overfitting_rmse:.4f}")
    
    if overfitting_r2 < 0.1:
        print("✅ Model shows good generalization (low overfitting)")
    else:
        print("⚠️  Model may be overfitting (consider regularization)")
    
    # 10. Summary
    print(f"\n🎯 FINAL SUMMARY")
    print("-" * 30)
    print(f"Dataset: California Housing ({len(house_price_dataframe)} samples)")
    print(f"Features: {len(X.columns)}")
    print(f"Model: XGBoost Regressor")
    print(f"Best Test R²: {test_r2:.4f}")
    print(f"Best Test RMSE: {test_rmse:.4f}")
    
    if test_r2 > 0.8:
        print("🎉 Excellent model performance!")
    elif test_r2 > 0.7:
        print("👍 Good model performance!")
    elif test_r2 > 0.6:
        print("👌 Acceptable model performance!")
    else:
        print("🔧 Model needs improvement!")
    
    print("\n" + "=" * 50)
    print("🏁 Analysis Complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check if all required libraries are installed:")
        print("pip install numpy pandas matplotlib seaborn scikit-learn xgboost") 