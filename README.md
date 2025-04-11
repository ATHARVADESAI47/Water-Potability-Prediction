# 💧 Water Potability Prediction Web App

This project is a **Streamlit-based Machine Learning web application** that predicts whether water is **potable (safe for drinking)** based on various chemical and physical parameters. The model is trained using a **Gradient Boosting Classifier** and includes several engineered features to enhance accuracy.

## 🚀 Features

- ✅ Predicts potability using user-input parameters
- 📈 Displays prediction confidence score
- 📊 Shows feature importance for model transparency
- 🧪 Reference table for safe water quality ranges
- 🔬 Includes feature engineering for improved performance

---

## 🧠 Technologies Used

- **Python 3**
- **Streamlit**
- **Scikit-learn**
- **XGBoost**
- **Pandas, NumPy**
- **Joblib** (for model & scaler persistence)

---

## 📊 Input Parameters

The app requires the following water quality features:

- pH
- Hardness
- Total Dissolved Solids (TDS)
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

Derived features such as ratios and deviations are also computed internally.

---

## 🌐 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/ATHARVADESAI47/Water-Potability-Prediction.git
   cd Water-Potability-Prediction
   pip install -r requirements.txt
   streamlit run app.py
📌 Dataset Reference
The dataset used is water_potability.csv, which contains information about water quality parameters and labels indicating whether the water is potable or not.

📬 Contact
Created by Atharva Desai
www.linkedin.com/in/atharva-desai-b266b3255
Feel free to connect or suggest improvements!


