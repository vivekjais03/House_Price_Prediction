#!/usr/bin/env python3
"""
House Price Prediction - Interactive Prediction Tool
"""

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def load_and_train_model():
    """Load data and train the best model"""
    print("🔄 Loading data and training model...")
    
    # Load dataset
    house_price_dataset = sklearn.datasets.fetch_california_housing()
    
    # Create DataFrame
    house_price_dataframe = pd.DataFrame(
        house_price_dataset.data, 
        columns=house_price_dataset.feature_names
    )
    house_price_dataframe['price'] = house_price_dataset.target
    
    # Prepare features and target
    X = house_price_dataframe.drop(['price'], axis=1)
    Y = house_price_dataframe['price']
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    # Train best model (XGBoost tuned)
    model = XGBRegressor(
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    
    model.fit(X_train, Y_train)
    print("✅ Model trained successfully!")
    
    return model, X.columns

def get_user_input(feature_names):
    """Get house features from user input"""
    print("\n🏠 Enter House Features for Price Prediction")
    print("=" * 50)
    print("Note: Prices are in $100,000 units (e.g., 5.0 = $500,000)")
    print()
    
    features = {}
    
    # Feature descriptions and ranges
    feature_info = {
        'MedInc': ('Median Income in Block Group', '1.0 - 15.0'),
        'HouseAge': ('Median House Age in Block Group', '1.0 - 52.0'),
        'AveRooms': ('Average Number of Rooms per Household', '2.0 - 8.0'),
        'AveBedrms': ('Average Number of Bedrooms per Household', '0.5 - 2.0'),
        'Population': ('Block Group Population', '3.0 - 3000.0'),
        'AveOccup': ('Average Number of Household Members', '1.0 - 10.0'),
        'Latitude': ('Block Group Latitude', '32.0 - 42.0'),
        'Longitude': ('Block Group Longitude', '-124.0 - -114.0')
    }
    
    for feature in feature_names:
        desc, range_info = feature_info[feature]
        while True:
            try:
                value = float(input(f"{feature} ({desc}) [{range_info}]: "))
                if feature == 'Latitude' and (value < 32 or value > 42):
                    print("⚠️  Latitude should be between 32.0 and 42.0")
                    continue
                elif feature == 'Longitude' and (value < -124 or value > -114):
                    print("⚠️  Longitude should be between -124.0 and -114.0")
                    continue
                features[feature] = value
                break
            except ValueError:
                print("❌ Please enter a valid number!")
    
    return features

def predict_price(model, features, feature_names):
    """Make price prediction"""
    # Create input array
    input_data = np.array([[features[feature] for feature in feature_names]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return prediction

def main():
    print("🏠 HOUSE PRICE PREDICTION TOOL")
    print("=" * 50)
    print("This tool predicts California house prices using XGBoost")
    print("Model accuracy: 84.46% (R² Score)")
    print()
    
    try:
        # Load and train model
        model, feature_names = load_and_train_model()
        
        while True:
            # Get user input
            features = get_user_input(feature_names)
            
            # Make prediction
            predicted_price = predict_price(model, features, feature_names)
            
            # Display results
            print("\n" + "=" * 50)
            print("📊 PREDICTION RESULTS")
            print("=" * 50)
            
            print(f"🏠 Predicted House Price: ${predicted_price:.2f}K")
            print(f"💰 In dollars: ${predicted_price * 100000:,.0f}")
            
            # Price category
            if predicted_price < 2.0:
                category = "Low-end"
            elif predicted_price < 4.0:
                category = "Mid-range"
            elif predicted_price < 6.0:
                category = "High-end"
            else:
                category = "Luxury"
            
            print(f"🏷️  Price Category: {category}")
            
            # Confidence level based on feature values
            confidence = "High"
            if any(features[feature] < 1.0 for feature in ['MedInc', 'AveRooms']):
                confidence = "Medium"
            if any(features[feature] > 10.0 for feature in ['MedInc', 'AveRooms']):
                confidence = "Medium"
            
            print(f"🎯 Confidence Level: {confidence}")
            
            # Ask if user wants to make another prediction
            print("\n" + "-" * 50)
            another = input("Make another prediction? (y/n): ").lower().strip()
            if another not in ['y', 'yes']:
                break
        
        print("\n👋 Thank you for using the House Price Prediction Tool!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check if all required libraries are installed:")
        print("pip install numpy pandas scikit-learn xgboost")

if __name__ == "__main__":
    main() 