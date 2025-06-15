import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model, scaler, and feature columns
model = joblib.load('life_expectancy_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load dataset for EDA (use same CSV as training)
df = pd.read_csv('Life Expectancy Data.csv')
df.columns = df.columns.str.strip()  # Clean column names

# Select only required columns
df = df[feature_columns + ['Life expectancy']].dropna()

# Set page config
st.set_page_config(page_title="Life Expectancy Project", page_icon="🌎", layout="centered")

# -------------------------------------
# 🟢 INTRODUCTION
# -------------------------------------
st.title("🌍 Life Expectancy Prediction App")
st.markdown("""
Welcome to the **Life Expectancy Prediction App**!  
This project explores global health and education factors that influence how long people live.

**Dataset:** WHO + UN sources  
**Goal:** Predict life expectancy using a simplified model based on just 5 features.
""")

# -------------------------------------
# 🟡 EDA SECTION
# -------------------------------------
st.header("📊 Exploratory Data Analysis")

with st.expander("🔍 View Summary Statistics"):
    st.write(df.describe())

with st.expander("📈 Feature Distributions"):
    feature_to_plot = st.selectbox("Select a feature to visualize:", feature_columns)
    fig, ax = plt.subplots()
    sns.histplot(df[feature_to_plot], kde=True, ax=ax, color='skyblue')
    st.pyplot(fig)

with st.expander("🎯 Correlation Heatmap"):
    # Encode 'Status' if it's still a string
    df_corr = df.copy()
    if df_corr['Status'].dtype == object:
        df_corr['Status'] = df_corr['Status'].map({'Developing': 0, 'Developed': 1})

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)


# -------------------------------------
# 🔵 MODEL SECTION
# -------------------------------------
st.header("🤖 Predict Life Expectancy")

st.markdown("""
This model uses a **Linear Regression algorithm** trained on the following 5 features:

- 🌐 Country Status (`Developed` or `Developing`)  
- 💀 Adult Mortality  
- 🍷 Alcohol Consumption  
- ⚖️ BMI  
- 🎓 Schooling

Enter your values below to get a real-time prediction:
""")

# Input layout
col1, col2 = st.columns(2)
user_input = []

for i, col in enumerate(feature_columns):
    if col == 'Status':
        with col1:
            val = st.selectbox("🌐 Country Status", [1, 0], format_func=lambda x: "Developed" if x == 1 else "Developing")
    elif col in ['Adult Mortality', 'Alcohol']:
        with col1:
            val = st.number_input(f"{col}", min_value=0.0, step=1.0)
    else:
        with col2:
            val = st.number_input(f"{col}", min_value=0.0, step=0.1)
    user_input.append(val)

# Predict
if st.button("🔮 Predict Life Expectancy"):
    input_df = pd.DataFrame([user_input], columns=feature_columns)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    st.success(f"🎯 Predicted Life Expectancy: `{prediction[0]:.2f} years`")

# -------------------------------------
# 🟣 CONCLUSION
# -------------------------------------
st.header("✅ Conclusion")
st.markdown("""
- Our simplified model performs reasonably well, with an **R² of ~0.72** using just 5 features.
- Key influencing factors include **education**, **healthcare**, and **economic development**.
- Predictions may vary by region, but the model gives valuable insights on population health.

---
*Project by Muhammad Osama Khan*  
*Built using Streamlit, Scikit-learn, Pandas & Matplotlib.*
""")