import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Universal Linear Regression App", layout="wide")

st.title("Universal Linear Regression App")
st.write(
    "Upload any CSV dataset, preprocess it automatically, "
    "train a Linear Regression model, and make predictions."
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# ---------------- MAIN PROCESS ----------------
if uploaded_file is not None:

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------- DATASET INFO ----------------
    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Rows:", df.shape[0])

    with col2:
        st.write("Columns:", df.shape[1])

    # ---------------- DATA TYPES ----------------
    st.subheader("Column Data Types")
    st.write(df.dtypes)

    # ---------------- MISSING VALUES ----------------
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # ---------------- HANDLE MISSING VALUES ----------------
    st.subheader("Handling Missing Values")

    missing_option = st.selectbox(
        "Choose Missing Value Handling Method",
        [
            "Drop Missing Values",
            "Fill Numeric with Mean",
            "Fill Numeric with Median"
        ]
    )

    if missing_option == "Drop Missing Values":
        df = df.dropna()

    elif missing_option == "Fill Numeric with Mean":
        for col in df.columns:
            if df[col].dtype != "object":
                df[col] = df[col].fillna(df[col].mean())

    elif missing_option == "Fill Numeric with Median":
        for col in df.columns:
            if df[col].dtype != "object":
                df[col] = df[col].fillna(df[col].median())

    st.success("Missing values handled successfully!")

    # ---------------- ENCODE CATEGORICAL COLUMNS ----------------
    st.subheader("Encoding Categorical Columns")

    label_encoders = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:

        st.write("Categorical Columns Found:")
        st.write(categorical_cols.tolist())

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        st.success("Categorical columns encoded successfully!")

    else:
        st.info("No categorical columns found.")

    # ---------------- SELECT TARGET COLUMN ----------------
    st.subheader("Select Target Column")

    target_column = st.selectbox(
        "Choose Target Variable",
        df.columns
    )

    # ---------------- FEATURES & TARGET ----------------
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    st.subheader("Feature Columns")
    st.write(X.columns.tolist())

    # ---------------- TRAIN TEST SPLIT ----------------
    st.subheader("Train-Test Split")

    test_size = st.slider(
        "Select Test Size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    # ---------------- MODEL TRAINING ----------------
    st.subheader("Model Training")

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.success("Model trained successfully!")

    # ---------------- PREDICTIONS ----------------
    y_pred = model.predict(X_test)

    # ---------------- MODEL EVALUATION ----------------
    st.subheader("Model Evaluation")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")

    with col2:
        st.metric("MAE", f"{mae:.4f}")
        st.metric("R² Score", f"{r2:.4f}")

    # ---------------- MODEL COEFFICIENTS ----------------
    st.subheader("Feature Importance / Coefficients")

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    st.dataframe(coef_df)

    # ---------------- CORRELATION HEATMAP ----------------
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # ---------------- ACTUAL VS PREDICTED ----------------
    st.subheader("Actual vs Predicted")

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.scatter(y_test, y_pred)

    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.set_title("Actual vs Predicted")

    st.pyplot(fig2)

    # ---------------- CUSTOM PREDICTION ----------------
    st.subheader("Make Custom Prediction")

    user_input = {}

    for column in X.columns:

        value = st.number_input(
            f"Enter value for {column}",
            value=float(X[column].mean())
        )

        user_input[column] = value

    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)

    st.success(f"Predicted Value: {prediction[0]:.4f}")

    # ---------------- DOWNLOAD CLEANED DATA ----------------
    st.subheader("Download Processed Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name="processed_dataset.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV dataset to continue.")