import streamlit as st
import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")

# Title
st.title("🏠 California House Price Prediction (XGBoost)")

# Load dataset
@st.cache_data
def load_data():
    house_data = sklearn.datasets.fetch_california_housing()
    df = pd.DataFrame(house_data.data, columns=house_data.feature_names)
    df["price"] = house_data.target
    return df

df = load_data()

# Sidebar
st.sidebar.header("Model Settings")
n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100, 50)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

# Train model
@st.cache_resource
def train_model(n_estimators, random_state):
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = XGBRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics_dict = {
        "train_r2": metrics.r2_score(y_train, y_pred_train),
        "train_rmse": np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)),
        "test_r2": metrics.r2_score(y_test, y_pred_test),
        "test_rmse": np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
    }

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, metrics_dict, feature_importance

model, metrics_dict, feature_importance = train_model(n_estimators, random_state)

# Metrics Display
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Train R²", f"{metrics_dict['train_r2']:.4f}")
    st.metric("Train RMSE", f"{metrics_dict['train_rmse']:.4f}")
with col2:
    st.metric("Test R²", f"{metrics_dict['test_r2']:.4f}")
    st.metric("Test RMSE", f"{metrics_dict['test_rmse']:.4f}")

# Feature Importance Plot
st.subheader("🔍 Feature Importance")
fig, ax = plt.subplots()
ax.barh(feature_importance["Feature"], feature_importance["Importance"])
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.invert_yaxis()
st.pyplot(fig)

# Prediction Section
st.subheader("🎯 Make a Prediction")
user_input = {}
for feature in df.columns[:-1]:
    user_input[feature] = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

if st.button("Predict House Price"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ${prediction*100000:.2f}")

st.caption("Data source: California Housing dataset (scikit-learn)")
