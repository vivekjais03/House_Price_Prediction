# 🏠 House Price Prediction

A comprehensive project for predicting housing prices using machine learning—with support for both notebooks and Python scripts. Includes a modular structure and optional web interface.

---

## 📌 Project Overview

Leverage the power of ML to predict housing prices using the **California Housing Dataset** from `scikit-learn`. This project includes:

- **Notebook version** (`House_Price_Prediction_Fixed.ipynb`)
- **Script-based version** (`house_price_prediction.py`, `improved_model.py`, `make_predictions.py`)
- **Optional web integration** via a `web/` submodule (if included)

---

## 🚀 Key Features

- **Data Processing**: Clean and prepare the dataset with Pandas.
- **Exploratory Analysis**: Visualize data through correlation heatmaps, scatter plots, and more using Matplotlib & Seaborn.
- **Modeling**: Train an **XGBoost Regressor** for high-accuracy predictions.
- **Evaluation**: Use R² and Mean Absolute Error (MAE) to quantify model performance.
- **Multiple Interfaces**:
  - Jupyter Notebook for interactive analysis.
  - Standalone Python scripts for automation and deployment.
- **Web Integration (Optional)**: A `web/` submodule if you'd like to build a front-end interface.

---

## 🗂 Repository Structure

```
House_Price_Prediction/
│
├── House_Price_Prediction_Fixed.ipynb   # Updated notebook
├── house_price_prediction.py            # Basic script version
├── improved_model.py                    # Refined modeling script
├── make_predictions.py                  # Script for making predictions
├── requirements.txt                     # Python dependencies
├── web/                                 # (Optional) Web interface as submodule
├── LICENSE                              # MIT License
└── README.md                            # Project documentation
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/vivekjais03/House_Price_Prediction.git
cd House_Price_Prediction
git submodule update --init --recursive  # If you plan to use the web interface
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Notebook  
- Use **Jupyter Notebook**, **VS Code**, or **Google Colab** to open and execute `House_Price_Prediction_Fixed.ipynb`.

### 4️⃣ Run the Scripts  
```bash
python house_price_prediction.py
python improved_model.py
python make_predictions.py
```

### 5️⃣ Web Interface (Optional)  
If using the `web/` submodule, follow its README to launch the front-end.

---

## 📊 Sample Results

*(Insert your actual results here after running the model)*

- **Train R² Score**: 0.95  
- **Train MAE**: 0.23  
- **Test R² Score**: 0.82  
- **Test MAE**: 0.35  

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 🙏 Acknowledgements

Special thanks to:
- Python libraries: NumPy, Pandas, scikit-learn, XGBoost, Matplotlib, Seaborn
- Mentor **Siddharshan** (YouTube) for guidance 
