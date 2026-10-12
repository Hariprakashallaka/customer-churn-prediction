
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------
# 1. Configure the Streamlit page
# -----------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"


# -----------------------------------------
# 2. Load the saved model
# -----------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to predict whether "
    "the customer is likely to churn."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run train_model.py first "
        "to create churn_model.pkl."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# -----------------------------------------
# 3. Collect customer information
# -----------------------------------------
# These columns and their order must match
# the dataset used in train_model.py.

with st.form("customer_form"):

    st.subheader("Customer Information")

    age = st.number_input(
        "Age", min_value=18, max_value=100, value=30
    )

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

    tenure = st.number_input(
        "Tenure", min_value=0, max_value=100,
        value=12
    )

    usage_frequency = st.number_input(
        "Usage Frequency", min_value=0,
        max_value=100, value=20
    )

    support_calls = st.number_input(
        "Support Calls", min_value=0,
        max_value=100, value=2
    )

    payment_delay = st.number_input(
        "Payment Delay", min_value=0,
        max_value=365, value=5
    )

    subscription_type = st.selectbox(
        "Subscription Type",
        ["Basic", "Standard", "Premium"],
    )

    contract_length = st.selectbox(
        "Contract Length",
        ["Monthly", "Quarterly", "Annual"],
    )

    total_spend = st.number_input(
        "Total Spend", min_value=0.0,
        value=1000.0, step=100.0
    )

    last_interaction = st.number_input(
        "Last Interaction",
        min_value=0, max_value=365, value=5
    )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True,
    )


# -----------------------------------------
# 4. Make the prediction
# -----------------------------------------

if submitted:

    # The column names must match the training data.
    customer_data = pd.DataFrame(
        [{  # Placeholder; not used in prediction
            "Age": age,
            "Gender": gender,
            "Tenure": tenure,
            "Usage Frequency": usage_frequency,
            "Support Calls": support_calls,
            "Payment Delay": payment_delay,
            "Subscription Type": subscription_type,
            "Contract Length": contract_length,
            "Total Spend": total_spend,
            "Last Interaction": last_interaction,
        }]
    )

    try:
        prediction = model.predict(customer_data)[0]

        st.subheader("Prediction Result")

        # Supports common binary labels: 0/1 or strings.
        if prediction == 1 or str(prediction).strip().lower() in (
            "yes", "true", "churn", "churned"
        ):
            st.error("⚠️ Prediction: Customer is likely to churn.")
        else:
            st.success("✅ Prediction: Customer is unlikely to churn.")

        # Display probability when the model supports it.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer_data)[0]
            classes = list(model.classes_)

            positive_class = next(
                (
                    cls for cls in classes
                    if str(cls).strip().lower()
                    in ("1", "yes", "true", "churn", "churned")
                ),
                None,
            )

            if positive_class is not None:
                churn_probability = probabilities[
                    classes.index(positive_class)
                ]

                st.metric(
                    "Estimated Churn Probability",
                    f"{churn_probability:.1%}",
                )

                st.caption(
                    "This is a model estimate, not a guarantee "
                    "of future customer behavior."
                )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}\n\n"
            "Check that the input columns, categories, and "
            "data types match your training dataset."
        )