import os
import json
import tarfile
import tempfile

import boto3
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


st.set_page_config(page_title="Loan Default Risk App", layout="wide")
st.title("Loan Default Risk Prediction")
st.write("Enter a few borrower details. The app fills the remaining model fields from a sample training row so the endpoint receives the full pipeline input.")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
file_path = os.path.join(project_root, "Portfolio", "X_train.csv")

X_train_sample = pd.read_csv(file_path)
X_train_sample = X_train_sample.loc[:, ~X_train_sample.columns.str.contains("^Unnamed")]

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource
def get_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session()
sm_session = sagemaker.Session(boto_session=session)

@st.cache_resource
def load_local_pipeline_from_s3():
    s3 = session.client("s3")
    local_tar = os.path.join(tempfile.gettempdir(), "finalized_loan_model.tar.gz")
    s3.download_file(aws_bucket, "sklearn-pipeline-deployment/finalized_loan_model.tar.gz", local_tar)
    extract_dir = tempfile.mkdtemp()
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(extract_dir)
    return joblib.load(os.path.join(extract_dir, "finalized_loan_model.joblib"))

pipeline = load_local_pipeline_from_s3()

important_inputs = []
for candidate in ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_avg", "term_months", "installment", "revol_util"]:
    if candidate in X_train_sample.columns:
        important_inputs.append(candidate)
important_inputs = important_inputs[:5]

st.subheader("Borrower Inputs")
with st.form("loan_form"):
    user_inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(important_inputs):
        values = pd.to_numeric(X_train_sample[col], errors="coerce")
        default = float(values.median()) if values.notna().any() else 0.0
        min_value = float(values.quantile(0.01)) if values.notna().any() else 0.0
        max_value = float(values.quantile(0.99)) if values.notna().any() else max(default + 1, 1.0)
        if min_value == max_value:
            max_value = min_value + 1.0
        with cols[i % 2]:
            user_inputs[col] = st.number_input(
                col.replace("_", " ").title(),
                min_value=min_value,
                max_value=max_value,
                value=default,
                step=(max_value - min_value) / 100
            )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    input_row = X_train_sample.iloc[[0]].copy()
    for col, value in user_inputs.items():
        input_row.loc[input_row.index[0], col] = value

    predictor = Predictor(
        endpoint_name=aws_endpoint,
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    response = predictor.predict(input_row.to_dict(orient="records"))
    pred = response.get("prediction", [None])[0]
    prob = response.get("probability_default", [None])[0]

    label = "Likely Default / Charged Off" if pred == 1 else "Likely Fully Paid"
    st.metric("Prediction", label)
    if prob is not None:
        st.metric("Estimated Default Probability", f"{prob:.1%}")

    display_explanation(input_df, session, aws_bucket)
        
    st.subheader("Decision Transparency")
    try:
        import shap
        preprocessor = pipeline.named_steps["preprocessor"]
        selector = pipeline.named_steps.get("selector", None)
        model = pipeline.named_steps["model"]

        X_transformed = preprocessor.transform(input_row)
        feature_names = preprocessor.get_feature_names_out()
        if selector is not None:
            X_transformed = selector.transform(X_transformed)
            feature_names = feature_names[selector.get_support()]

        if hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            importance = np.zeros(len(feature_names))

        explanation_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(10)

        st.write("Top factors used by the model for this type of prediction:")
        st.dataframe(explanation_df)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(explanation_df["Feature"][::-1], explanation_df["Importance"][::-1])
        ax.set_title("Top Model Factors")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Explanation plot could not be displayed: {e}")
