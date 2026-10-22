
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = "models/credit_risk_model.pkl"
FEATURES_PATH = "models/features.pkl"
MODEL_NAME_PATH = "models/model_name.pkl"
DATA_PATH = "archive/customer_data.csv"

st.markdown("""
<style>
.stApp {
    background: #f4f7fc;
}
h1 {
    color: #183153;
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

@st.cache_data
def load_customers():
    return pd.read_csv(DATA_PATH)

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("Model files are missing. Run python train_model.py first.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Customer dataset not found in the archive folder.")
    st.stop()

model, features = load_model()
customers = load_customers()

model_name = (
    joblib.load(MODEL_NAME_PATH)
    if os.path.exists(MODEL_NAME_PATH)
    else "Credit Risk Model"
)

st.title("💳 Credit Risk Classification AI")
st.caption("Machine Learning Based Customer Credit Risk Assessment")

dashboard, predict, about = st.tabs([
    "📊 Dashboard",
    "🔍 Predict Customer Risk",
    "ℹ️ About Project"
])

with dashboard:
    st.subheader("📊 Credit Risk Analytics")
    st.write(
        "Explore customer risk distribution and the performance "
        "of the trained machine learning model."
    )

    # Prepare dataset
    df = customers.copy()
    df["Risk Category"] = df["label"].map({
        0: "Low Risk",
        1: "High Risk"
    })

    total_customers = len(df)
    low_risk = int((df["label"] == 0).sum())
    high_risk = int((df["label"] == 1).sum())

    low_pct = low_risk / total_customers * 100 if total_customers else 0
    high_pct = high_risk / total_customers * 100 if total_customers else 0

    # KPI cards
    st.markdown("### Customer Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Low Risk", f"{low_risk:,}", f"{low_pct:.1f}%")
    c3.metric("High Risk", f"{high_risk:,}", f"{high_pct:.1f}%")
    c4.metric("Model Features", len(features))

    st.divider()

    # Risk distribution charts
    left, right = st.columns(2)

    with left:
        st.markdown("### Risk Distribution")

        risk_counts = pd.DataFrame({
            "Customer Count": [low_risk, high_risk]
        }, index=["Low Risk", "High Risk"])

        st.bar_chart(
            risk_counts,
            color=["#4F9D69"],
            height=300
        )

        st.caption("Number of customers in each actual dataset category.")

    with right:
        st.markdown("### Risk Composition")

        risk_composition = pd.DataFrame({
            "Percentage": [low_pct, high_pct]
        }, index=["Low Risk", "High Risk"])

        st.bar_chart(
            risk_composition,
            color=["#6488E8"],
            height=300
        )

        st.caption("Percentage of customers belonging to each category.")

    st.divider()

    # Model evaluation results from the latest training run
    st.markdown("### 🤖 Model Performance")

    st.caption(
        "Metrics below are from the test-set evaluation in your latest "
        "training run. Retrain and update these values if the test "
        "split or model changes."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy", "63.56%")
    m2.metric("ROC-AUC", "0.7036")
    m3.metric("High-Risk Recall", "71.11%")

    performance = pd.DataFrame({
        "Score (%)": [63.56, 70.36, 71.11]
    }, index=["Accuracy", "ROC-AUC × 100", "High-Risk Recall"])

    st.bar_chart(performance, color="#8267C8", height=260)

    st.divider()

    # Confusion matrix
    st.markdown("### Confusion Matrix")

    confusion = pd.DataFrame(
        [[111, 69], [13, 32]],
        index=["Actual Low Risk", "Actual High Risk"],
        columns=["Predicted Low Risk", "Predicted High Risk"]
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )

    st.caption(
        "Rows represent actual categories; columns represent model predictions. "
        "These counts come from the latest 225-customer test set."
    )

    st.divider()

    # Dataset preview
    st.markdown("### Customer Dataset Preview")

    risk_filter = st.selectbox(
        "Filter customers by actual risk category",
        ["All Customers", "Low Risk", "High Risk"]
    )

    if risk_filter != "All Customers":
        filtered_df = df[df["Risk Category"] == risk_filter]
    else:
        filtered_df = df

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"Showing up to 100 of {len(filtered_df):,} matching customers."
    )


with predict:
    st.subheader("Predict Customer Risk")
    st.write(
        "Select a customer from the dataset. Their available features "
        "will be loaded automatically."
    )

    customer_ids = customers["id"].dropna().tolist()
    selected_id = st.selectbox(
        "Select Customer ID",
        customer_ids
    )

    customer_row = customers.loc[
        customers["id"] == selected_id
    ].iloc[0]
    
    
    # -------- DETAILED CUSTOMER RISK PROFILE --------
    st.divider()
    st.markdown("### 👤 Customer Risk Profile")

    payment_path = "archive/payment_data.csv"

    if os.path.exists(payment_path):
        payment_df = pd.read_csv(payment_path)

        # Match the selected customer to their payment records
        payment_df["id"] = pd.to_numeric(
            payment_df["id"], errors="coerce"
        )

        customer_payments = payment_df[
            payment_df["id"] == float(selected_id)
        ].copy()

        payment_columns = [
            "OVD_t1", "OVD_t2", "OVD_t3", "OVD_sum",
            "pay_normal", "prod_limit", "new_balance",
            "highest_balance"
        ]

        available_columns = [
            col for col in payment_columns
            if col in customer_payments.columns
        ]

        for col in available_columns:
            customer_payments[col] = pd.to_numeric(
                customer_payments[col], errors="coerce"
            )

        if not customer_payments.empty:
            st.caption(
                f"Payment records found: {len(customer_payments)}"
            )

            # Summary metrics
            def safe_sum(column):
                if column not in customer_payments.columns:
                    return 0.0
                return customer_payments[column].sum()

            def safe_mean(column):
                if column not in customer_payments.columns:
                    return 0.0
                result = customer_payments[column].mean()
                return float(result) if pd.notna(result) else 0.0

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Overdue Amount (OVD Sum)",
                f"{safe_sum('OVD_sum'):,.2f}"
            )
            p2.metric(
                "Normal Payments",
                f"{safe_sum('pay_normal'):,.0f}"
            )
            p3.metric(
                "Average New Balance",
                f"{safe_mean('new_balance'):,.2f}"
            )
            p4.metric(
                "Highest Recorded Balance",
                f"{customer_payments['highest_balance'].max():,.2f}"
                if "highest_balance" in customer_payments.columns
                and customer_payments["highest_balance"].notna().any()
                else "N/A"
            )

            st.markdown("#### Payment Behaviour")

            # Overdue history chart
            overdue_columns = [
                col for col in ["OVD_t1", "OVD_t2", "OVD_t3"]
                if col in customer_payments.columns
            ]

            if overdue_columns:
                overdue_summary = pd.DataFrame({
                    "Average Overdue Value": [
                        customer_payments[col].mean()
                        for col in overdue_columns
                    ]
                }, index=overdue_columns)

                st.markdown("**Average overdue values by period**")
                st.bar_chart(
                    overdue_summary,
                    color="#E76F51"
                )

            # Balance comparison
            balance_columns = [
                col for col in ["prod_limit", "new_balance",
                                "highest_balance"]
                if col in customer_payments.columns
            ]

            if balance_columns:
                balance_summary = pd.DataFrame({
                    "Average Value": [
                        customer_payments[col].mean()
                        for col in balance_columns
                    ]
                }, index=balance_columns)

                st.markdown("**Account balance overview**")
                st.bar_chart(
                    balance_summary,
                    color="#6488E8"
                )

            # Individual payment records
            with st.expander("View detailed payment records"):
                st.dataframe(
                    customer_payments.drop(columns=["id"], errors="ignore"),
                    use_container_width=True,
                    hide_index=True
                )

            # Simple descriptive profile (not a causal explanation)
            overdue_total = safe_sum("OVD_sum")
            normal_total = safe_sum("pay_normal")

            st.markdown("#### Profile Summary")

            if overdue_total > 0:
                st.warning(
                    "The customer's records contain a positive aggregate "
                    "overdue value. Review the payment history alongside "
                    "the model's prediction."
                )
            else:
                st.info(
                    "The aggregate overdue value is zero or unavailable. "
                    "This alone does not establish that the customer is "
                    "low risk."
                )

            st.write(
                f"Across the available payment records, the aggregate "
                f"normal-payment value is {normal_total:,.2f}. "
                "These are descriptive dataset statistics, not proof "
                "of future repayment behaviour."
            )

        else:
            st.info(
                "No payment records were found for this customer ID."
            )
    else:
        st.warning("Payment history file was not found.")

    st.markdown("#### Customer Information")

    a, b = st.columns(2)
    a.metric("Customer ID", str(selected_id))

    if "label" in customer_row.index:
        actual = int(customer_row["label"])
        a_label = "High Risk" if actual == 1 else "Low Risk"
        b.metric("Actual Dataset Label", a_label)

    st.caption(
        "The actual label is shown for project evaluation only. "
        "The model prediction is calculated separately."
    )

    if st.button("🔍 Predict Credit Risk", use_container_width=True):
        input_data = pd.DataFrame(
            [[customer_row.get(f, float("nan")) for f in features]],
            columns=features
        )

        # Add aggregated payment features in the same way as training.
        payment_path = "archive/payment_data.csv"

        if os.path.exists(payment_path):
            payment = pd.read_csv(payment_path)

            payment_features = [
                "OVD_t1", "OVD_t2", "OVD_t3", "OVD_sum",
                "pay_normal", "prod_limit", "new_balance",
                "highest_balance"
            ]

            available = [
                c for c in payment_features if c in payment.columns
            ]

            if "id" in payment.columns and available:
                for col in available:
                    payment[col] = pd.to_numeric(
                        payment[col], errors="coerce"
                    )

                agg = payment.groupby("id")[available].agg(
                    ["sum", "mean", "max"]
                )
                agg.columns = [
                    f"{col}_{stat}" for col, stat in agg.columns
                ]

                if selected_id in agg.index:
                    for feature in features:
                        if feature in agg.columns:
                            input_data.loc[0, feature] = agg.loc[
                                selected_id, feature
                            ]

        prediction = int(model.predict(input_data)[0])

        st.markdown("### Prediction Result")

        if prediction == 1:
            st.error("⚠️ Model Prediction: HIGH RISK")
        else:
            st.success("✅ Model Prediction: LOW RISK")

        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            high_idx = classes.index(1)
            probability = model.predict_proba(input_data)[0][high_idx]

            st.metric(
                "Estimated High-Risk Probability",
                f"{probability * 100:.2f}%"
            )
            st.progress(float(probability))

        st.warning(
            "Educational prototype only. Predictions are not suitable "
            "for real-world lending decisions."
        )

with about:
    st.subheader("About This Project")
    st.markdown("""
    **Objective:** Classify customers into low-risk and high-risk
    categories using historical customer and payment information.

    **Technologies**
    - Python
    - Pandas
    - Scikit-learn
    - Logistic Regression and Random Forest
    - Streamlit
    - Joblib

    **Workflow**
    1. Load customer and payment datasets.
    2. Aggregate payment history by customer.
    3. Handle missing values.
    4. Train and evaluate classification models.
    5. Save the selected model.
    6. Predict credit-risk categories through a web interface.

    **Target labels:** 0 = Low Risk, 1 = High Risk.
    """)
