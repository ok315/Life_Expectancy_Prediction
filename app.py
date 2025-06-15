import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------
# 🎨 Page Configuration & Styling
# -------------------------------
st.set_page_config(page_title="Life Expectancy App", page_icon="🌎", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f0f0f0 !important;
        }
        .main {
            background-color: #f0f0f0 !important;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #2a8cff;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        .css-1v0mbdj.edgvbvh3, .css-1v0mbdj.ef3psqc12 {
            font-size: 18px !important;
            margin-bottom: 1rem !important;
        }
        section[data-testid="stSidebar"] {
            float: right;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Load model, scaler, and data
# ------------------------------
model = joblib.load('life_expectancy_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load dataset
df = pd.read_csv('Life Expectancy Data.csv')
df.columns = df.columns.str.strip()
df = df[feature_columns + ['Life expectancy']].dropna()

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["📌 Introduction", "📊 EDA", "🤖 Predict", "✅ Conclusion"])

# ------------------------------
# 📌 Introduction
# ------------------------------
if page == "📌 Introduction":
    st.title("🌍 Life Expectancy Prediction App")
    st.markdown("""
    Welcome to the **Life Expectancy Prediction App**! 🌎

    This project is developed as part of an academic assignment to explore how machine learning can provide meaningful insights from global health data.

    Using real-world statistics from the **World Health Organization (WHO)** and the **United Nations**, this app allows users to predict life expectancy based on a set of key features such as adult mortality, alcohol consumption, BMI, country status, and average years of schooling.

    Life expectancy is a crucial indicator of a nation’s health and quality of life. It reflects the cumulative effect of various socio-economic and medical factors, and is widely used by researchers and policymakers to guide development strategies.

    This app includes:
    - 📊 Interactive exploratory data analysis
    - 🤖 A machine learning model trained with **Linear Regression**
    - 🧪 Real-time predictions based on user input
    - 📈 Model performance metrics for evaluation

    Whether you're a data science student, a researcher, or just curious, this tool offers a hands-on demonstration of applying ML to real-world challenges.
    """)

# ------------------------------
# 📊 EDA
# ------------------------------
elif page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")

    with st.expander("🔍 Summary Statistics"):
        st.write(df.describe())

    with st.expander("📈 Feature Distributions"):
        feature_to_plot = st.selectbox("Select a feature to visualize:", feature_columns)
        fig, ax = plt.subplots()
        sns.histplot(df[feature_to_plot], kde=True, ax=ax, color='skyblue')
        st.pyplot(fig)

    with st.expander("🎯 Correlation Heatmap"):
        df_corr = df.copy()
        if df_corr['Status'].dtype == object:
            df_corr['Status'] = df_corr['Status'].map({'Developing': 0, 'Developed': 1})

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# ------------------------------
# 🤖 Predict
# ------------------------------
elif page == "🤖 Predict":
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

    if st.button("🔮 Predict Life Expectancy"):
        input_df = pd.DataFrame([user_input], columns=feature_columns)
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        st.success(f"🎯 Predicted Life Expectancy: `{prediction[0]:.2f} years`")

# ------------------------------
# ✅ Conclusion
# ------------------------------
elif page == "✅ Conclusion":
    st.header("✅ Conclusion")
    st.markdown("""
    This project demonstrated how health and education-related features can be used to predict life expectancy using a simple linear regression model.

    **Key Takeaways:**
    - 🔑 Education and healthcare access play a vital role in increasing life expectancy.
    - 🧠 A model trained with just 5 features can still offer meaningful insights.
    - 🌐 Streamlit helps build powerful and interactive ML apps with ease.
    - 📊 Through EDA and modeling, we discovered strong correlations between mortality, education, and development level with life expectancy.
    - 📈 Despite being a simple model, our regression approach achieved an R² score of ~0.72 and an RMSE of ~4.85 years.

    ---
    #### 🤝 Connect With Me
    <a href='https://www.linkedin.com/in/muhammad-osama-khan-4573a6295/' target='_blank'><button style='background-color:#0e76a8;color:white;padding:0.5rem 1rem;border:none;border-radius:10px;margin-right:1rem;'>Connect on LinkedIn</button></a>
    <a href='https://github.com/ok315' target='_blank'><button style='background-color:#333;color:white;padding:0.5rem 1rem;border:none;border-radius:10px;'>Follow on GitHub</button></a>

    <br>
    <small>Project by <strong>Muhammad Osama Khan</strong> – BS Data Science, PUCIT</small>
    """, unsafe_allow_html=True)
