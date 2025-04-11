import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load pre-trained model and scaler
try:
    model = joblib.load('gradient_boosting_model1.pkl')
    scaler = joblib.load('scaler1.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {str(e)}")
    st.stop()

st.title("💧 Water Potability Prediction")
st.subheader("Enter the Water Quality Parameters")

col1, col2 = st.columns(2)

with col1:
    ph = st.slider("pH Level (0-14)", 0.0, 14.0, 7.0)
    hardness = st.slider("Hardness (mg/L)", 47.0, 323.0, 196.0)
    solids = st.slider("Total Dissolved Solids (mg/L)", 500.0, 50000.0, 22000.0)
    chloramines = st.slider("Chloramines (ppm)", 0.35, 13.13, 7.12)
    sulfate = st.slider("Sulfate (mg/L)", 129.0, 481.0, 333.0)

with col2:
    conductivity = st.slider("Conductivity (μS/cm)", 180.0, 750.0, 426.0)
    organic_carbon = st.slider("Organic Carbon (ppm)", 2.2, 28.3, 14.3)
    trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.7, 124.0, 66.0)
    turbidity = st.slider("Turbidity (NTU)", 1.45, 6.74, 3.97)

# Derived features
tds_to_conductivity = solids / conductivity if conductivity != 0 else 0
organic_to_turbidity = organic_carbon / turbidity if turbidity != 0 else 0
hardness_to_solids = hardness / solids if solids != 0 else 0
ph_deviation = abs(ph - 7.0)
tds_concentration = solids / conductivity if conductivity != 0 else 0
organic_load = organic_carbon * turbidity

# Feature array
features = np.array([[ph, hardness, solids, chloramines, sulfate,
                      conductivity, organic_carbon, trihalomethanes, turbidity,
                      tds_to_conductivity, organic_to_turbidity, hardness_to_solids,
                      ph_deviation, tds_concentration, organic_load]])

if st.button("🔍 Check Potability"):
    try:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)

        st.subheader("🧪 Prediction Result:")
        if prediction[0] == 1:
            st.success(f"✅ Water is Potable ({probability[0][1]*100:.2f}% confidence)")
        else:
            st.error(f"❌ Water is Not Potable ({probability[0][0]*100:.2f}% confidence)")

        # Feature Importance
        st.subheader("📊 Feature Importance")
        feature_names = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                         'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity',
                         'TDS/Conductivity', 'Organic/Turbidity', 'Hardness/Solids',
                         'pH Deviation', 'TDS Concentration', 'Organic Load']
        importances = model.feature_importances_

        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        st.bar_chart(df_importance.set_index('Feature'))

        # Parameter info
        st.subheader("📚 Parameter Reference")
        ref_df = pd.DataFrame({
            'Parameter': feature_names[:9],
            'Current Value': [ph, hardness, solids, chloramines, sulfate,
                              conductivity, organic_carbon, trihalomethanes, turbidity],
            'Safe Range': [
                '6.5–8.5', '47–323 mg/L', '500–1000 mg/L', '0.35–4 ppm', '3–250 mg/L',
                '180–400 μS/cm', '0–2 ppm', '0–80 μg/L', '0–5 NTU'
            ]
        })
        st.dataframe(ref_df)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
