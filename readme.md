🌍 Life Expectancy Prediction App  
A streamlined machine learning application that predicts life expectancy using global health and education indicators, built with Linear Regression and Streamlit.

---

📦 Dataset Used:  
Life Expectancy (WHO, UN stats)  
Link of Dataset: "https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who"


📊 Features

- **Real-time Life Expectancy Prediction**  
  Predict life expectancy using 5 simplified indicators: development status, adult mortality, alcohol, BMI, and schooling.

- **Interactive Data Exploration**  
  Summary statistics, feature distributions, and correlation heatmaps for better understanding of the dataset.

- **Model Evaluation**  
  View model performance using R² score and RMSE (Root Mean Squared Error) for regression quality.

- **User-Friendly Interface**  
  Simple and clean input form with real-time prediction output and visual explanation.

---

📱 Application Sections

### 🟢 Introduction
- Overview of the problem and dataset source
- Objective of the app: Predict life expectancy using limited features

### 🟡 EDA (Exploratory Data Analysis)
- Interactive visualizations:
  - Feature histograms with KDE
  - Correlation heatmap
  - Summary statistics
- Missing value and data distribution insights

### 🔵 Model Section
- Model: **Linear Regression**
- Input features:
  - Status (Developed / Developing)
  - Adult Mortality
  - Alcohol
  - BMI
  - Schooling
- Evaluation:
  - R² Score ≈ 0.72
  - RMSE ≈ 4.85 years

### 🔮 Prediction
- Interactive prediction form
- Users can input feature values and get:
  - Instant life expectancy prediction
  - Clean UI with emojis and tooltips

### ✅ Conclusion
- Key factors affecting life expectancy identified
- Simpler models can still perform well with core indicators
- Open for future improvements (more features, better models)

---

🛠️ Technologies Used

- **Streamlit** – Web application interface
- **Pandas & NumPy** – Data wrangling and numerical processing
- **Scikit-learn** – Model training and evaluation
- **Matplotlib & Seaborn** – Visualizations
- **Joblib** – Saving/loading model and scalers

---

📁 Project Structure

| File                         | Description                                  |
|------------------------------|----------------------------------------------|
| `app.py`                     | Main Streamlit app                           |
| `life_expectancy.csv`        | Dataset used                                 |
| `life_expectancy_model.pkl`  | Trained model file                           |
| `scaler.pkl`                 | Feature scaler                               |
| `feature_columns.pkl`        | Column order during training                 |
| `Life_Expectancy_Analysis.ipynb` | Notebook for EDA + model training       |

---

👨‍💻 Author  
**Muhammad Osama Khan**  
BS Data Science – PUCIT  
GitHub: [github.com/ok315](https://github.com/ok315)

---

📌 Note  
This project showcases EDA, model development, evaluation, and interactive deployment.