import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="centered")

st.title("Machine Learning Assignment 2")
#st.write("Bank Term Deposit Subscription Prediction")

# -----------------------------
# Load trained models
# -----------------------------
models = {
    "Logistic Regression": joblib.load("model/model_lr.pkl"),
    "Decision Tree": joblib.load("model/model_dt.pkl"),
    "KNN": joblib.load("model/model_knn.pkl"),
    "Naive Bayes": joblib.load("model/model_nb.pkl"),
    "Random Forest": joblib.load("model/model_rf.pkl"),
    "XGBoost": joblib.load("model/model_xgb.pkl")
}

# -----------------------------
# Upload test dataset
# -----------------------------
st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (use test_data.csv)",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    # -----------------------------
    # Separate features and target
    # -----------------------------
    if "deposit" not in data.columns:
        st.error("Target column 'deposit' not found in the dataset.")
    else:
        X = data.drop("deposit", axis=1)
        y = data["deposit"]

        # -----------------------------
        # Model selection
        # -----------------------------
        st.subheader("Select Model")
        selected_model = st.selectbox(
            "Choose a Machine Learning Model",
            list(models.keys())
        )

        if st.button("Run Model"):
            model = models[selected_model]

            # Prediction
            y_pred = model.predict(X)

            # -----------------------------
            # Metrics output
            # -----------------------------
            st.subheader("Classification Report")
            report = classification_report(y, y_pred, output_dict=False)
            st.text(report)

            # -----------------------------
            # Confusion Matrix
            # -----------------------------
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
