from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import sklearn.datasets
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json

app = FastAPI(title="House Price Prediction API", version="1.0.0")

# Allow local dev frontend
app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:3000",
    "https://house-price-prediction-pi-two.vercel.app"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

class PredictRequest(BaseModel):
	MedInc: float = Field(..., description="Median income in block group")
	HouseAge: float = Field(..., description="Median house age in block group")
	AveRooms: float = Field(..., description="Average rooms per household")
	AveBedrms: float = Field(..., description="Average bedrooms per household")
	Population: float = Field(..., description="Block group population")
	AveOccup: float = Field(..., description="Average household members")
	Latitude: float = Field(..., description="Latitude")
	Longitude: float = Field(..., description="Longitude")

class PredictResponse(BaseModel):
	predicted_price_100k: float
	predicted_price_usd: float

# Train model on startup
feature_names = [
	"MedInc",
	"HouseAge",
	"AveRooms",
	"AveBedrms",
	"Population",
	"AveOccup",
	"Latitude",
	"Longitude",
]

model: XGBRegressor | None = None
training_data: pd.DataFrame | None = None
test_predictions: np.ndarray | None = None
test_actual: np.ndarray | None = None

@app.on_event("startup")
def train_model():
	global model, training_data, test_predictions, test_actual
	housing = sklearn.datasets.fetch_california_housing()
	df = pd.DataFrame(housing.data, columns=housing.feature_names)
	df["price"] = housing.target
	
	X = df[feature_names]
	Y = df["price"]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
	
	model = XGBRegressor(
		random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1,
		subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
	)
	model.fit(X_train, Y_train)
	
	# Store data for charts
	training_data = df
	test_predictions = model.predict(X_test)
	test_actual = Y_test

@app.get("/")
def root():
	return {"status": "ok", "message": "House Price Prediction API"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
	if model is None:
		raise RuntimeError("Model not ready")
	values = np.array([[getattr(req, name) for name in feature_names]])
	pred_100k = float(model.predict(values)[0])
	return PredictResponse(
		predicted_price_100k=round(pred_100k, 4),
		predicted_price_usd=round(pred_100k * 100000, 2),
	)

@app.get("/chart-data/feature-importance")
def get_feature_importance():
	if model is None:
		raise RuntimeError("Model not ready")
	
	importance = model.feature_importances_
	feature_importance_data = [
		{"feature": feature, "importance": float(importance[i])}
		for i, feature in enumerate(feature_names)
	]
	
	# Sort by importance
	feature_importance_data.sort(key=lambda x: x["importance"], reverse=True)
	
	return {
		"labels": [item["feature"] for item in feature_importance_data],
		"data": [item["importance"] for item in feature_importance_data]
	}

@app.get("/chart-data/price-distribution")
def get_price_distribution():
    if training_data is None:
        raise RuntimeError("Training data not ready")
    
    # Sample 1000 random prices for better visualization
    sample_prices = training_data["price"].sample(n=min(1000, len(training_data)), random_state=42)
    
    return {
        "prices": [float(price * 100000) for price in sample_prices.tolist()],  # Convert to USD and list
        "labels": ["House Prices (USD)"]
    }

@app.get("/chart-data/feature-vs-price")
def get_feature_vs_price():
    if training_data is None:
        raise RuntimeError("Training data not ready")
    
    # Sample 500 data points for better visualization
    sample_data = training_data.sample(n=min(500, len(training_data)), random_state=42)
    
    # Get top 3 most important features
    if model is not None:
        importance = model.feature_importances_
        top_features_idx = np.argsort(importance)[-3:][::-1]
        top_features = [feature_names[i] for i in top_features_idx]
    else:
        top_features = ["MedInc", "HouseAge", "AveRooms"]
    
    charts_data = {}
    for feature in top_features:
        # Convert to lists and ensure proper indexing
        feature_values = sample_data[feature].tolist()
        price_values = sample_data["price"].tolist()
        
        charts_data[feature] = {
            "x": [float(val) for val in feature_values],
            "y": [float(price * 100000) for price in price_values],  # Convert to USD
            "xLabel": feature,
            "yLabel": "Price (USD)"
        }
    
    return charts_data

@app.get("/chart-data/model-performance")
def get_model_performance():
    if test_predictions is None or test_actual is None:
        raise RuntimeError("Test data not ready")
    
    # Calculate metrics
    r2 = r2_score(test_actual, test_predictions)
    mae = mean_absolute_error(test_actual, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
    
    # Sample 200 points for prediction vs actual chart
    sample_size = min(200, len(test_actual))
    indices = np.random.choice(len(test_actual), sample_size, replace=False)
    
    return {
        "metrics": {
            "r2_score": float(r2),
            "mae": float(mae),
            "rmse": float(rmse)
        },
        "predictions_vs_actual": {
            "actual": [float(test_actual[int(i)] * 100000) for i in indices],  # Convert to USD and fix indexing
            "predicted": [float(test_predictions[int(i)] * 100000) for i in indices],  # Convert to USD and fix indexing
            "labels": ["Actual Price (USD)", "Predicted Price (USD)"]
        }
    }

@app.get("/chart-data/sample-data")
def get_sample_data():
    if training_data is None:
        raise RuntimeError("Training data not ready")
    
    # Return sample data for initial chart display
    sample = training_data.sample(n=min(100, len(training_data)), random_state=42)
    
    return {
        "features": {
            "MedInc": [float(val) for val in sample["MedInc"].tolist()],
            "HouseAge": [float(val) for val in sample["HouseAge"].tolist()],
            "AveRooms": [float(val) for val in sample["AveRooms"].tolist()],
            "AveBedrms": [float(val) for val in sample["AveBedrms"].tolist()],
            "Population": [float(val) for val in sample["Population"].tolist()],
            "AveOccup": [float(val) for val in sample["AveOccup"].tolist()],
            "Latitude": [float(val) for val in sample["Latitude"].tolist()],
            "Longitude": [float(val) for val in sample["Longitude"].tolist()]
        },
        "prices": [float(price * 100000) for price in sample["price"].tolist()]  # Convert to USD and list
    } 

@app.post("/personalized-analytics")
def get_personalized_analytics(request: PredictRequest):
    if model is None or training_data is None:
        raise RuntimeError("Model not ready")
    
    # Get user's input values
    user_input = {
        "MedInc": request.MedInc,
        "HouseAge": request.HouseAge,
        "AveRooms": request.AveRooms,
        "AveBedrms": request.AveBedrms,
        "Population": request.Population,
        "AveOccup": request.AveOccup,
        "Latitude": request.Latitude,
        "Longitude": request.Longitude
    }
    
    # Make prediction for user's data
    user_values = np.array([[getattr(request, name) for name in feature_names]])
    user_prediction = float(model.predict(user_values)[0])
    user_price_usd = user_prediction * 100000
    
    # Find similar houses in the dataset (within 20% of each feature)
    similar_houses = []
    for idx, row in training_data.iterrows():
        is_similar = True
        for feature in feature_names:
            user_val = user_input[feature]
            dataset_val = row[feature]
            # Check if within 20% range
            if abs(user_val - dataset_val) / max(abs(user_val), 0.1) > 0.2:
                is_similar = False
                break
        if is_similar:
            similar_houses.append({
                "price": float(row["price"] * 100000),  # Convert to USD
                "features": {f: float(row[f]) for f in feature_names}
            })
    
    # If not enough similar houses, use broader criteria
    if len(similar_houses) < 10:
        similar_houses = []
        for idx, row in training_data.iterrows():
            # Use 50% range for broader comparison
            is_similar = True
            for feature in feature_names:
                user_val = user_input[feature]
                dataset_val = row[feature]
                if abs(user_val - dataset_val) / max(abs(user_val), 0.1) > 0.5:
                    is_similar = False
                    break
            if is_similar:
                similar_houses.append({
                    "price": float(row["price"] * 100000),
                    "features": {f: float(row[f]) for f in feature_names}
                })
    
    # Get price distribution for similar houses
    similar_prices = [house["price"] for house in similar_houses] if similar_houses else []
    
    # Calculate percentile of user's predicted price
    if similar_prices:
        similar_prices.sort()
        user_percentile = sum(1 for p in similar_prices if p <= user_price_usd) / len(similar_prices) * 100
    else:
        user_percentile = 50  # Default if no similar houses
    
    # Feature comparison: how user's features compare to dataset averages
    dataset_averages = {}
    for feature in feature_names:
        dataset_averages[feature] = float(training_data[feature].mean())
    
    feature_comparison = {}
    for feature in feature_names:
        user_val = user_input[feature]
        avg_val = dataset_averages[feature]
        percentage_diff = ((user_val - avg_val) / avg_val) * 100 if avg_val != 0 else 0
        
        feature_comparison[feature] = {
            "user_value": user_val,
            "dataset_average": avg_val,
            "percentage_difference": percentage_diff,
            "status": "above" if percentage_diff > 0 else "below" if percentage_diff < 0 else "average"
        }
    
    # Price sensitivity analysis: how changing each feature affects price
    price_sensitivity = {}
    base_prediction = user_prediction
    
    for feature in feature_names:
        # Test +10% change in feature
        test_values = user_values.copy()
        current_val = user_input[feature]
        test_values[0][feature_names.index(feature)] = current_val * 1.1
        
        new_prediction = float(model.predict(test_values)[0])
        price_change = (new_prediction - base_prediction) / base_prediction * 100
        
        price_sensitivity[feature] = {
            "feature": feature,
            "price_change_percent": price_change,
            "sensitivity": "high" if abs(price_change) > 5 else "medium" if abs(price_change) > 2 else "low"
        }
    
    return {
        "user_prediction": {
            "predicted_price_100k": round(user_prediction, 4),
            "predicted_price_usd": round(user_price_usd, 2),
            "percentile": round(user_percentile, 1)
        },
        "similar_houses": {
            "count": len(similar_houses),
            "price_range": {
                "min": round(min(similar_prices), 2) if similar_prices else 0,
                "max": round(max(similar_prices), 2) if similar_prices else 0,
                "average": round(sum(similar_prices) / len(similar_prices), 2) if similar_prices else 0
            },
            "prices": similar_prices[:100]  # Limit to 100 for chart
        },
        "feature_comparison": feature_comparison,
        "price_sensitivity": price_sensitivity,
        "dataset_context": {
            "total_houses": len(training_data),
            "dataset_averages": dataset_averages
        }
    } 
