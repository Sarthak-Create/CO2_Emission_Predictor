import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="CO₂ Emissions Predictor", layout="wide")
st.title("🚗 CO₂ Emissions Prediction App")

# === Upload CSV ===
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # === Data Cleaning ===
    df.drop_duplicates(inplace=True)
    df.drop(['Make', 'Model', 'Vehicle Class'], axis=1, errors='ignore', inplace=True)
    
    if 'Fuel Type' in df.columns:
        le = LabelEncoder()
        df['Fuel Type'] = le.fit_transform(df['Fuel Type'])

    # === Visualizations ===
    with st.expander("🔍 Show Exploratory Data Analysis"):
        st.subheader("CO₂ Emissions Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['CO2 Emissions(g/km)'], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

    # === Feature Selection ===
    features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
                'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
                'Fuel Type']
    target = 'CO2 Emissions(g/km)'

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Model Selection ===
    model_name = st.selectbox("Choose a model", ["Linear Regression", "Random Forest", "XGBoost"])

    if st.button("🔮 Predict and Evaluate"):
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        else:
            model = XGBRegressor(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # === Metrics ===
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.success(f"✅ Model Trained: {model_name}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # === Plot Actual vs Predicted ===
        st.subheader("📈 Actual vs Predicted CO₂ Emissions")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(y_test.values[:50], label='Actual', marker='o')
        ax3.plot(y_pred[:50], label='Predicted', marker='x')
        ax3.set_title("Actual vs Predicted")
        ax3.legend()
        st.pyplot(fig3)
