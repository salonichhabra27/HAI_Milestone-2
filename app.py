import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Fair Income Prediction Model",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL AND RESULTS
# -------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("income_prediction_model.pkl")

@st.cache_data
def load_data():
    overall = pd.read_csv("overall_metrics.csv")
    fairness = pd.read_csv("fairness_metrics.csv")
    subgroup = pd.read_csv("subgroup_metrics.csv")
    return overall, fairness, subgroup


model = load_model()
overall_df, fairness_df, subgroup_df = load_data()


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🤖 Fair Income Prediction System")

st.write(
    """
    This application demonstrates a machine learning model trained on the
    UCI Adult Census Income dataset. The model predicts whether an individual's
    income is likely to be above or below $50K per year based on selected
    demographic and employment-related attributes.
    """
)

st.warning(
    """
    **Responsible Use Notice:** This is an educational demonstration.
    The prediction is a statistical estimate based on historical data and
    should not be used as the sole basis for employment, financial, legal,
    or other high-stakes decisions.
    """
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Make a Prediction",
        "📊 Performance & Fairness",
        "🔍 Global Explainability",
        "ℹ️ Model Information & Limitations",
        "🧠 HAX Evaluation"
    ]
)


# =================================================
# PAGE 1: MAKE A PREDICTION
# =================================================

if page == "🏠 Make a Prediction":

    st.header("Make a Prediction")

    st.write(
        """
        Enter the information below. The final model does not directly use
        **sex or race** as input features because these sensitive attributes
        were removed as part of the bias mitigation strategy.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            min_value=17,
            max_value=90,
            value=35
        )

        workclass = st.selectbox(
            "Work Class",
            [
                "Private",
                "Self-emp-not-inc",
                "Self-emp-inc",
                "Federal-gov",
                "Local-gov",
                "State-gov",
                "Without-pay",
                "Never-worked"
            ]
        )

        education = st.selectbox(
            "Education",
            [
                "HS-grad",
                "Some-college",
                "Bachelors",
                "Masters",
                "Assoc-voc",
                "Assoc-acdm",
                "11th",
                "9th",
                "7th-8th",
                "Prof-school",
                "Doctorate",
                "5th-6th",
                "10th",
                "1st-4th",
                "Preschool",
                "12th"
            ]
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Never-married",
                "Married-civ-spouse",
                "Divorced",
                "Separated",
                "Widowed",
                "Married-spouse-absent",
                "Married-AF-spouse"
            ]
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Tech-support",
                "Craft-repair",
                "Other-service",
                "Sales",
                "Exec-managerial",
                "Prof-specialty",
                "Handlers-cleaners",
                "Machine-op-inspct",
                "Adm-clerical",
                "Farming-fishing",
                "Transport-moving",
                "Priv-house-serv",
                "Protective-serv",
                "Armed-Forces"
            ]
        )

    with col2:

        relationship = st.selectbox(
            "Relationship",
            [
                "Wife",
                "Own-child",
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Unmarried"
            ]
        )

        capital_gain = st.number_input(
            "Capital Gain",
            min_value=0,
            value=0
        )

        capital_loss = st.number_input(
            "Capital Loss",
            min_value=0,
            value=0
        )

        hours_per_week = st.slider(
            "Hours per Week",
            min_value=1,
            max_value=99,
            value=40
        )

        native_country = st.selectbox(
            "Native Country",
            [
                "United-States",
                "Mexico",
                "Philippines",
                "Germany",
                "Canada",
                "India",
                "England",
                "China",
                "Japan",
                "Other"
            ]
        )


    st.divider()

    if st.button("🔮 Predict Income", type="primary"):

        education_num_map = {
            "Preschool": 1,
            "1st-4th": 2,
            "5th-6th": 3,
            "7th-8th": 4,
            "9th": 5,
            "10th": 6,
            "11th": 7,
            "12th": 8,
            "HS-grad": 9,
            "Some-college": 10,
            "Assoc-voc": 11,
            "Assoc-acdm": 12,
            "Bachelors": 13,
            "Masters": 14,
            "Prof-school": 15,
            "Doctorate": 16
        }

        input_data = pd.DataFrame({
            "age": [age],
            "workclass": [workclass],
            "education": [education],
            "education_num": [education_num_map[education]],
            "marital_status": [marital_status],
            "occupation": [occupation],
            "relationship": [relationship],
            "capital_gain": [capital_gain],
            "capital_loss": [capital_loss],
            "hours_per_week": [hours_per_week],
            "native_country": [native_country]
        })

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.success("### Predicted Income: More than $50K")
            else:
                st.info("### Predicted Income: $50K or less")

            st.metric(
                "Estimated probability of income > $50K",
                f"{probability:.1%}"
            )

            st.caption(
                """
                The probability represents the model's statistical confidence
                based on patterns in the training data. It should not be
                interpreted as a guarantee or a measure of an individual's
                ability or potential.
                """
            )
            # -------------------------------------------------
            # LOCAL EXPLANATION
            # -------------------------------------------------

            st.subheader("🔎 Why did the model make this prediction?")

            preprocessor = model.named_steps["prep"]
            classifier = model.named_steps["model"]

            # Transform the single user input using the same preprocessing
            transformed_input = preprocessor.transform(input_data)

            # Feature names after preprocessing
            feature_names = preprocessor.get_feature_names_out()

            # Convert sparse matrix to dense array if needed
            if hasattr(transformed_input, "toarray"):
                transformed_values = transformed_input.toarray()[0]
            else:
                transformed_values = transformed_input[0]

            # Logistic regression coefficients
            coefficients = classifier.coef_[0]

            # Local contribution = transformed feature value × coefficient
            contributions = transformed_values * coefficients

            local_df = pd.DataFrame({
                "Feature": feature_names,
                "Contribution": contributions
            })

            local_df["Feature"] = (
                local_df["Feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )

            # Keep the features with the strongest absolute contribution
            local_df["Absolute Contribution"] = local_df["Contribution"].abs()

            top_local = (
                local_df
                .sort_values("Absolute Contribution", ascending=False)
                .head(8)
            )

            st.write(
                """
                The chart below shows which features had the strongest
                influence on this specific prediction.

                Positive contributions push the prediction toward
                **income > $50K**, while negative contributions push it
                toward **$50K or less**.
                """
            )

            st.bar_chart(
                top_local.set_index("Feature")["Contribution"]
            )

            st.dataframe(
                top_local[["Feature", "Contribution"]],
                use_container_width=True,
                hide_index=True
            )

            positive_local = (
                top_local[top_local["Contribution"] > 0]
                .sort_values("Contribution", ascending=False)
            )

            negative_local = (
                top_local[top_local["Contribution"] < 0]
                .sort_values("Contribution", ascending=True)
            )

            if not positive_local.empty:
                st.write(
                    "**Main factors pushing toward income > $50K:** "
                    + ", ".join(positive_local["Feature"].head(3).tolist())
                )

            if not negative_local.empty:
                st.write(
                    "**Main factors pushing toward income ≤ $50K:** "
                    + ", ".join(negative_local["Feature"].head(3).tolist())
                )

            st.info(
                """
                This is a local explanation for the current input only.
                It describes how the model combined the entered features
                for this prediction. It does not mean these factors cause
                a person's real-world income.
                """
            )

        

        except Exception as e:
            st.error(f"Prediction error: {e}")

            st.info(
                """
                Check that the input features in this interface match the
                features used to train the saved model.
                """
            )   

    


# =================================================
# PAGE 2: PERFORMANCE AND FAIRNESS
# =================================================

elif page == "📊 Performance & Fairness":

    st.header("📊 Model Performance and Fairness")

    st.subheader("Overall Model Performance")

    st.write(
        """
        The table below compares model performance before and after
        bias mitigation.
        """
    )

    st.dataframe(
        overall_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("⚖️ Fairness Metrics")

    st.write(
        """
        Fairness was evaluated before and after mitigation using
        demographic parity and equal opportunity related measures.
        Lower differences generally indicate more similar outcomes
        across demographic groups.
        """
    )

    st.dataframe(
        fairness_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("Subgroup Performance")

    st.write(
        """
        Overall accuracy alone can hide differences between groups.
        The following results show model performance across demographic
        and intersectional subgroups.
        """
    )

    st.dataframe(
        subgroup_df,
        use_container_width=True
    )


# =================================================
# PAGE 3: GLOBAL EXPLAINABILITY
# =================================================

elif page == "🔍 Global Explainability":

    st.header("🔍 Global Model Explainability")

    st.write(
        """
        Global explanations describe how the model behaves overall rather
        than explaining only one individual prediction.

        Because the final model is a Logistic Regression model, its learned
        coefficients can be used to understand which features generally
        push predictions toward income > $50K and which push predictions
        toward $50K or less.
        """
    )

    try:
        # Access preprocessing and classifier
        preprocessor = model.named_steps["prep"]
        classifier = model.named_steps["model"]
        

        # Get transformed feature names and coefficients
        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients
        })

        # Clean names for easier reading
        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
        )

        positive_features = (
            importance_df
            .sort_values("Coefficient", ascending=False)
            .head(10)
        )

        negative_features = (
            importance_df
            .sort_values("Coefficient", ascending=True)
            .head(10)
        )

        st.subheader("⬆️ Features Associated with Income > $50K")

        st.write(
            "Positive coefficients push the model toward predicting income above $50K."
        )

        st.bar_chart(
            positive_features.set_index("Feature")["Coefficient"]
        )

        st.dataframe(
            positive_features,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("⬇️ Features Associated with Income ≤ $50K")

        st.write(
            "Negative coefficients push the model toward predicting income of $50K or less."
        )

        negative_display = negative_features.copy()
        negative_display["Magnitude"] = negative_display["Coefficient"].abs()

        st.bar_chart(
            negative_display.set_index("Feature")["Magnitude"]
        )

        st.dataframe(
            negative_features,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            """
            These coefficients show associations learned from historical
            training data. They should not be interpreted as evidence that
            a feature directly causes a person's income.
            """
        )

    except Exception as e:
        st.error(f"Unable to generate global explanation: {e}")





# =================================================
# PAGE 3: MODEL INFORMATION
# =================================================

elif page == "ℹ️ Model Information & Limitations":

    st.header("ℹ️ What This Model Can and Cannot Do")

    st.subheader("What the model does")

    st.write(
        """
        The model uses selected employment, education and economic
        attributes from the Adult Census Income dataset to estimate
        whether income is likely to be above or below $50K.
        """
    )

    st.subheader("Bias Mitigation Strategy")

    st.write(
        """
        The original baseline model included sensitive demographic
        attributes. To reduce unfairness, the project applied two
        mitigation strategies:
        """
    )

    st.markdown(
        """
        1. **Feature elimination:** Sex and race were removed from
           the predictive input features.

        2. **Intersectional reweighing:** Training examples from
           different demographic intersections were assigned weights
           to reduce imbalance and unequal representation.
        """
    )

    st.subheader("Important Limitations")

    st.warning(
        """
        - The model is trained on historical census data and may reflect
          historical social and economic inequalities.

        - Removing sensitive attributes does not guarantee complete fairness.
          Other variables may act as proxies.

        - Model predictions are probabilistic estimates and can be incorrect.

        - Performance may vary across demographic groups.

        - This system should not be used as the sole basis for high-stakes
          decisions such as hiring, lending, insurance, or legal decisions.
        """
    )


# =================================================
# PAGE 4: HAX EVALUATION
# =================================================

elif page == "🧠 HAX Evaluation":

    st.header("🧠 Human-AI Experience (HAX) Evaluation")

    st.write(
        """
        The interface was evaluated using principles inspired by
        Microsoft's Human-AI Experience Guidelines.
        """
    )

    hax_data = pd.DataFrame({
        "HAX Principle": [
            "Make clear what the system can do",
            "Make clear how well the system can do",
            "Support efficient correction",
            "Make uncertainty clear",
            "Set expectations and limitations",
            "Support responsible use"
        ],

        "How the Interface Addresses It": [
            "The Model Information page explains the prediction task and dataset.",
            "Overall performance, fairness metrics and subgroup results are displayed.",
            "Users can modify their inputs and immediately request a new prediction.",
            "The interface displays the estimated probability and explains that it is not a guarantee.",
            "A dedicated limitations section explains bias, historical data and model limitations.",
            "Warnings explain that the model should not be used as the sole basis for high-stakes decisions."
        ],

        "Area for Improvement": [
            "Could provide more examples of appropriate and inappropriate use.",
            "Could use more visual charts instead of only tables.",
            "Could allow users to compare multiple hypothetical profiles.",
            "Could provide confidence intervals or uncertainty explanations.",
            "Could include dataset coverage information for individual inputs.",
            "Could add individual prediction explanations such as feature importance."
        ]
    })

    st.dataframe(
        hax_data,
        use_container_width=True
    )

    st.subheader("Overall Reflection")

    st.write(
        """
        The interface meets several HAX principles by clearly communicating
        the model's purpose, performance, fairness characteristics,
        uncertainty and limitations. The strongest aspects are transparency
        and responsible-use communication.

        However, the current prototype could be improved with visual
        explanations, individual-level interpretability and more detailed
        uncertainty information.
        """
    )
