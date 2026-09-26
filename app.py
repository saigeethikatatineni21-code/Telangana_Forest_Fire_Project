import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import date, timedelta


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telangana Forest Fire Analysis",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(
        "telangana_forest_fire_dataset.csv"
    )

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce"
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["frp"] = pd.to_numeric(
        df["frp"],
        errors="coerce"
    )

    return df


@st.cache_resource
def load_models():

    severity_data = joblib.load(
        "fire_model.pkl"
    )

    # fire_model.pkl contains the model + metadata
    if isinstance(severity_data, dict):
        severity_model = severity_data["model"]
    else:
        severity_model = severity_data

    future_model = joblib.load(
        "future_fire_risk_model.pkl"
    )

    future_metadata = joblib.load(
        "future_risk_metadata.pkl"
    )

    return (
        severity_model,
        future_model,
        future_metadata
    )


@st.cache_data
def load_feature_importance():

    return pd.read_csv(
        "feature_importance.csv"
    )


@st.cache_data
def load_future_feature_importance():

    return pd.read_csv(
        "future_risk_feature_importance.csv"
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

df = load_dataset()

(
    severity_model,
    future_model,
    future_metadata
) = load_models()

feature_importance = load_feature_importance()

future_feature_importance = (
    load_future_feature_importance()
)


# ============================================================
# FEATURE IMPORTANCE HELPER
# ============================================================

def prepare_importance_dataframe(data):

    """
    Makes the feature-importance CSV compatible even if
    column names use different capitalization such as:

    importance / Importance
    feature / Feature
    """

    data = data.copy()

    # Remove accidental spaces from column names
    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    # Find feature column
    feature_column = None

    for column in data.columns:

        if str(column).lower() in [
            "feature",
            "features",
            "feature_name",
            "feature name"
        ]:

            feature_column = column
            break

    # Find importance column
    importance_column = None

    for column in data.columns:

        if str(column).lower() in [
            "importance",
            "feature_importance",
            "feature importance",
            "importance_score"
        ]:

            importance_column = column
            break

    if (
        feature_column is None
        or importance_column is None
    ):

        return None

    result = data[
        [
            feature_column,
            importance_column
        ]
    ].copy()

    result.columns = [
        "feature",
        "importance"
    ]

    result["importance"] = pd.to_numeric(
        result["importance"],
        errors="coerce"
    )

    result = result.dropna(
        subset=["importance"]
    )

    return result


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🔥 Telangana Forest Fire"
)

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Fire Map",
        "Current Severity Prediction",
        "Future Fire Risk",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "NASA FIRMS-based Telangana "
    "forest fire analysis and prediction system."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_grid_value(
    value,
    grid_size=0.25
):

    """
    Convert latitude/longitude into the same
    0.25-degree grid used by the future-risk model.
    """

    return np.floor(
        float(value) / grid_size
    ) * grid_size


def calculate_historical_features(
    latitude,
    longitude,
    prediction_date
):

    """
    Calculate historical features required by the
    future fire-risk Random Forest model.

    IMPORTANT:
    Only observations BEFORE the selected prediction date
    are used.
    """

    grid_size = 0.25

    grid_lat = get_grid_value(
        latitude,
        grid_size
    )

    grid_lon = get_grid_value(
        longitude,
        grid_size
    )

    prediction_date = pd.Timestamp(
        prediction_date
    )

    # --------------------------------------------------------
    # Prepare FIRMS data
    # --------------------------------------------------------

    working = df[
        [
            "acq_date",
            "latitude",
            "longitude",
            "frp"
        ]
    ].copy()

    working["grid_lat"] = (
        np.floor(
            working["latitude"]
            / grid_size
        )
        * grid_size
    )

    working["grid_lon"] = (
        np.floor(
            working["longitude"]
            / grid_size
        )
        * grid_size
    )

    # --------------------------------------------------------
    # Select requested grid and only past observations
    # --------------------------------------------------------

    grid_data = working[
        (
            np.isclose(
                working["grid_lat"],
                grid_lat
            )
        )
        &
        (
            np.isclose(
                working["grid_lon"],
                grid_lon
            )
        )
        &
        (
            working["acq_date"]
            < prediction_date
        )
    ].copy()

    # --------------------------------------------------------
    # Check historical data
    # --------------------------------------------------------

    if grid_data.empty:

        return None, (
            "No historical FIRMS observations were found "
            "for this 0.25° grid cell before the selected date."
        )

    # --------------------------------------------------------
    # Create 30-day historical window
    # --------------------------------------------------------

    start_date = (
        prediction_date
        - timedelta(days=30)
    )

    date_range = pd.date_range(
        start=start_date,
        end=(
            prediction_date
            - timedelta(days=1)
        ),
        freq="D"
    )

    # --------------------------------------------------------
    # Daily fire count
    # --------------------------------------------------------

    daily_count = (
        grid_data
        .groupby("acq_date")
        .size()
        .reindex(
            date_range,
            fill_value=0
        )
    )

    daily_count.index.name = "acq_date"

    # --------------------------------------------------------
    # Daily mean FRP
    # --------------------------------------------------------

    daily_frp = (
        grid_data
        .groupby("acq_date")["frp"]
        .mean()
        .reindex(
            date_range
        )
        .fillna(0)
    )

    daily_frp.index.name = "acq_date"

    # --------------------------------------------------------
    # Previous day
    # --------------------------------------------------------

    fire_lag_1 = float(
        daily_count.iloc[-1]
    )

    # --------------------------------------------------------
    # Previous 3 days
    # --------------------------------------------------------

    fire_lag_3 = float(
        daily_count.iloc[-3]
    )

    # --------------------------------------------------------
    # Previous 7 days
    # --------------------------------------------------------

    fire_lag_7 = float(
        daily_count.iloc[-7]
    )

    # --------------------------------------------------------
    # Previous 7-day fire count
    # --------------------------------------------------------

    fire_rolling_7 = float(
        daily_count.iloc[-7:].sum()
    )

    # --------------------------------------------------------
    # Previous 30-day fire count
    # --------------------------------------------------------

    fire_rolling_30 = float(
        daily_count.sum()
    )

    # --------------------------------------------------------
    # Previous 7-day average FRP
    # --------------------------------------------------------

    frp_rolling_7 = float(
        daily_frp.iloc[-7:].mean()
    )

    # --------------------------------------------------------
    # Previous 30-day average FRP
    # --------------------------------------------------------

    frp_rolling_30 = float(
        daily_frp.mean()
    )

    # --------------------------------------------------------
    # Fire-active days in previous 30 days
    # --------------------------------------------------------

    fire_days_30 = float(
        (daily_count > 0).sum()
    )

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    month = int(
        prediction_date.month
    )

    day_of_year = int(
        prediction_date.dayofyear
    )

    # --------------------------------------------------------
    # Model features
    # --------------------------------------------------------

    features = {

        "grid_lat":
            grid_lat,

        "grid_lon":
            grid_lon,

        "month":
            month,

        "day_of_year":
            day_of_year,

        "fire_lag_1":
            fire_lag_1,

        "fire_lag_3":
            fire_lag_3,

        "fire_lag_7":
            fire_lag_7,

        "fire_rolling_7":
            fire_rolling_7,

        "fire_rolling_30":
            fire_rolling_30,

        "frp_rolling_7":
            frp_rolling_7,

        "frp_rolling_30":
            frp_rolling_30,

        "fire_days_30":
            fire_days_30
    }

    feature_df = pd.DataFrame(
        [features]
    )

    return feature_df, None


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    probability
):

    if probability < 0.30:

        return "Low"

    elif probability < 0.60:

        return "Moderate"

    elif probability < 0.80:

        return "High"

    else:

        return "Very High"


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title(
        "🔥 Telangana Forest Fire Dashboard"
    )

    st.markdown(
        """
        ### NASA FIRMS-based Fire Analysis

        This dashboard analyzes satellite-based active-fire
        observations for Telangana and provides both current
        fire-severity prediction and next-day fire-risk estimation.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_fires = len(df)

    average_frp = df["frp"].mean()

    years = df["year"].nunique()

    high_fires = df[
        df["fire_severity"].isin(
            [
                "High",
                "Very High"
            ]
        )
    ].shape[0]

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Total Fire Detections",
            f"{total_fires:,}"
        )

    with col2:

        st.metric(
            "Average FRP",
            f"{average_frp:.2f}"
        )

    with col3:

        st.metric(
            "Years Covered",
            years
        )

    with col4:

        st.metric(
            "High / Very High",
            f"{high_fires:,}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # YEARLY FIRE DETECTIONS
    # --------------------------------------------------------

    st.subheader(
        "📊 Fire Detections by Year"
    )

    yearly = (
        df
        .groupby("year")
        .size()
        .reset_index(
            name="fire_count"
        )
    )

    st.bar_chart(
        yearly.set_index(
            "year"
        )
    )

    # --------------------------------------------------------
    # SEASONAL ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "🌦️ Seasonal Fire Distribution"
    )

    seasonal = (
        df
        .groupby("season")
        .size()
        .reset_index(
            name="fire_count"
        )
    )

    st.bar_chart(
        seasonal.set_index(
            "season"
        )
    )

    # --------------------------------------------------------
    # SEVERITY DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "🔥 Fire Severity Distribution"
    )

    severity_counts = (
        df["fire_severity"]
        .value_counts()
        .reset_index()
    )

    severity_counts.columns = [
        "Severity",
        "Count"
    ]

    st.dataframe(
        severity_counts,
        hide_index=True
    )


# ============================================================
# PAGE 2 — FIRE MAP
# ============================================================

elif page == "Fire Map":

    st.title(
        "🗺️ Telangana Fire Detection Map"
    )

    st.write(
        "Filter the satellite fire detections by year "
        "and severity."
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        selected_year = st.selectbox(
            "Select Year",
            sorted(
                df["year"]
                .dropna()
                .unique(),
                reverse=True
            )
        )

    with col2:

        severity_options = [
            "All",
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]

        selected_severity = st.selectbox(
            "Select Severity",
            severity_options
        )

    map_df = df[
        df["year"] == selected_year
    ].copy()

    if selected_severity != "All":

        map_df = map_df[
            map_df["fire_severity"]
            == selected_severity
        ]

    st.write(
        f"Showing {len(map_df):,} fire detections."
    )

    if not map_df.empty:

        map_data = map_df[
            [
                "latitude",
                "longitude"
            ]
        ].dropna()

        st.map(
            map_data
        )

    else:

        st.warning(
            "No fire detections match the selected filters."
        )


# ============================================================
# PAGE 3 — CURRENT FIRE SEVERITY PREDICTION
# ============================================================

elif page == "Current Severity Prediction":

    st.title(
        "🔥 Current Fire Severity Prediction"
    )

    st.write(
        """
        Enter satellite observation features to estimate
        the observed fire severity class.
        """
    )

    st.info(
        "FRP is not used as an input because it was used "
        "to create the severity target. This avoids target leakage."
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=15.0,
            max_value=21.0,
            value=18.11,
            step=0.01
        )

        longitude = st.number_input(
            "Longitude",
            min_value=76.0,
            max_value=83.0,
            value=79.25,
            step=0.01
        )

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=8,
            step=1
        )

        day_of_year = st.number_input(
            "Day of Year",
            min_value=1,
            max_value=366,
            value=268,
            step=1
        )

    with col2:

        bright_ti4 = st.number_input(
            "Brightness Temperature T(I4)",
            value=334.01,
            step=0.1
        )

        bright_ti5 = st.number_input(
            "Brightness Temperature T(I5)",
            value=304.00,
            step=0.1
        )

        scan = st.number_input(
            "Scan",
            min_value=0.0,
            value=0.46,
            step=0.01
        )

        track = st.number_input(
            "Track",
            min_value=0.0,
            value=0.47,
            step=0.01
        )

    if st.button(
        "🔍 Predict Current Severity",
        type="primary"
    ):

        input_data = pd.DataFrame(
            [{
                "latitude": latitude,
                "longitude": longitude,
                "month": month,
                "day_of_year": day_of_year,
                "bright_ti4": bright_ti4,
                "bright_ti5": bright_ti5,
                "scan": scan,
                "track": track
            }]
        )

        prediction = (
            severity_model
            .predict(
                input_data
            )[0]
        )

        probabilities = (
            severity_model
            .predict_proba(
                input_data
            )[0]
        )

        classes = (
            severity_model.classes_
        )

        probability_df = pd.DataFrame({

            "Severity":
                classes,

            "Probability":
                probabilities
        })

        st.success(
            f"Predicted Fire Severity: **{prediction}**"
        )

        st.subheader(
            "Prediction Probabilities"
        )

        probability_df[
            "Probability"
        ] = (
            probability_df[
                "Probability"
            ] * 100
        ).round(2)

        st.dataframe(
            probability_df,
            hide_index=True
        )

        st.bar_chart(
            probability_df.set_index(
                "Severity"
            )
        )


# ============================================================
# PAGE 4 — FUTURE FIRE RISK
# ============================================================

elif page == "Future Fire Risk":

    st.title(
        "🔮 Future Fire Risk Prediction"
    )

    st.markdown(
        """
        ### Predict next-day fire activity

        Enter a location and prediction date. The system
        automatically calculates recent fire activity and
        FRP-based historical features from the NASA FIRMS
        dataset before making the prediction.
        """
    )

    st.info(
        "The prediction estimates the probability of fire "
        "activity on the day after the selected date. "
        "It is not a guarantee that a fire will occur."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    st.subheader(
        "📍 Prediction Inputs"
    )

    col1, col2 = (
        st.columns(2)
    )

    with col1:

        future_latitude = st.number_input(
            "Latitude",
            min_value=15.0,
            max_value=21.0,
            value=18.11,
            step=0.01,
            key="future_latitude"
        )

    with col2:

        future_longitude = st.number_input(
            "Longitude",
            min_value=76.0,
            max_value=83.0,
            value=79.25,
            step=0.01,
            key="future_longitude"
        )

    # --------------------------------------------------------
    # FUTURE DATE FIX
    # --------------------------------------------------------
    #
    # We allow the user to select TODAY.
    #
    # If today is 2026-09-24:
    #
    # Selected date = 2026-09-24
    # Prediction target = 2026-09-25
    #
    # The model uses only observations BEFORE 2026-09-24.
    #
    # --------------------------------------------------------

    today = date.today()

    dataset_min_date = (
        df["acq_date"]
        .min()
    )

    if pd.isna(
        dataset_min_date
    ):

        st.error(
            "The dataset does not contain valid dates."
        )

        st.stop()

    min_prediction_date = (
        dataset_min_date
        + pd.Timedelta(days=30)
    ).date()

    # IMPORTANT:
    # Do NOT use df["acq_date"].max() as max date.
    #
    # The historical dataset can end earlier than today.
    # We still want to predict tomorrow using historical data.

    max_prediction_date = today

    # Make sure min <= max
    if (
        min_prediction_date
        > max_prediction_date
    ):

        st.error(
            "The dataset does not contain enough historical "
            "data to make a prediction for the current date."
        )

        st.stop()

    prediction_date = st.date_input(
        "Prediction Date",
        value=today,
        min_value=min_prediction_date,
        max_value=max_prediction_date
    )

    target_date = (
        pd.Timestamp(
            prediction_date
        )
        + pd.Timedelta(days=1)
    )

    st.caption(
        f"The model uses historical information before "
        f"{pd.Timestamp(prediction_date):%Y-%m-%d} "
        f"to estimate fire activity for "
        f"{target_date:%Y-%m-%d}."
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Next-Day Fire Risk",
        type="primary"
    ):

        with st.spinner(
            "Calculating historical fire-risk features..."
        ):

            (
                feature_df,
                error_message
            ) = calculate_historical_features(
                future_latitude,
                future_longitude,
                prediction_date
            )

        if error_message:

            st.error(
                error_message
            )

        else:

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            probability_array = (
                future_model
                .predict_proba(
                    feature_df
                )[0]
            )

            classes = list(
                future_model.classes_
            )

            if 1 in classes:

                positive_index = (
                    classes.index(1)
                )

                fire_probability = float(
                    probability_array[
                        positive_index
                    ]
                )

            else:

                fire_probability = 0.0

            prediction = (
                future_model
                .predict(
                    feature_df
                )[0]
            )

            risk_level = get_risk_level(
                fire_probability
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "🎯 Prediction Result"
            )

            st.success(
                f"Prediction target date: "
                f"**{target_date:%Y-%m-%d}**"
            )

            result_col1, result_col2 = (
                st.columns(2)
            )

            with result_col1:

                st.metric(
                    "Next-Day Fire Probability",
                    f"{fire_probability * 100:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Estimated Risk Level",
                    risk_level
                )

            if risk_level == "Low":

                st.success(
                    "🟢 Estimated Risk: LOW"
                )

            elif risk_level == "Moderate":

                st.warning(
                    "🟡 Estimated Risk: MODERATE"
                )

            elif risk_level == "High":

                st.warning(
                    "🟠 Estimated Risk: HIGH"
                )

            else:

                st.error(
                    "🔴 Estimated Risk: VERY HIGH"
                )

            # ------------------------------------------------
            # HISTORICAL FEATURES
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "📊 Historical Features Used"
            )

            feature_display = pd.DataFrame({

                "Feature": [

                    "Grid Latitude",

                    "Grid Longitude",

                    "Month",

                    "Day of Year",

                    "Fire Detections - Previous Day",

                    "Fire Detections - 3 Days Ago",

                    "Fire Detections - 7 Days Ago",

                    "Fire Count - Previous 7 Days",

                    "Fire Count - Previous 30 Days",

                    "Average FRP - Previous 7 Days",

                    "Average FRP - Previous 30 Days",

                    "Fire-Active Days - Previous 30 Days"
                ],

                "Value": [

                    feature_df.iloc[0][
                        "grid_lat"
                    ],

                    feature_df.iloc[0][
                        "grid_lon"
                    ],

                    feature_df.iloc[0][
                        "month"
                    ],

                    feature_df.iloc[0][
                        "day_of_year"
                    ],

                    feature_df.iloc[0][
                        "fire_lag_1"
                    ],

                    feature_df.iloc[0][
                        "fire_lag_3"
                    ],

                    feature_df.iloc[0][
                        "fire_lag_7"
                    ],

                    feature_df.iloc[0][
                        "fire_rolling_7"
                    ],

                    feature_df.iloc[0][
                        "fire_rolling_30"
                    ],

                    feature_df.iloc[0][
                        "frp_rolling_7"
                    ],

                    feature_df.iloc[0][
                        "frp_rolling_30"
                    ],

                    feature_df.iloc[0][
                        "fire_days_30"
                    ]
                ]
            })

            st.dataframe(
                feature_display,
                hide_index=True
            )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "ℹ️ Interpretation"
            )

            st.write(
                f"""
                For the selected location, the model estimates a
                **{fire_probability * 100:.2f}% probability of fire
                activity on {target_date:%Y-%m-%d}.

                The estimated risk category is **{risk_level}**.

                The prediction is based on historical NASA FIRMS
                fire activity in the corresponding 0.25-degree grid
                cell, including recent fire counts, recent FRP,
                seasonal timing and the number of fire-active days.
                """
            )

            st.caption(
                "Risk categories used by this application: "
                "0–30% Low, 30–60% Moderate, 60–80% High, "
                "80–100% Very High. These are application-defined "
                "presentation categories, not official government "
                "risk thresholds."
            )


# ============================================================
# PAGE 5 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "📈 Model Performance"
    )

    # ========================================================
    # CURRENT SEVERITY MODEL
    # ========================================================

    st.header(
        "1. Current Fire Severity Model"
    )

    st.write(
        "Random Forest Classifier"
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    with metric_col1:

        st.metric(
            "Accuracy",
            "81.19%"
        )

    with metric_col2:

        st.metric(
            "Precision",
            "82.05%"
        )

    with metric_col3:

        st.metric(
            "Recall",
            "81.19%"
        )

    with metric_col4:

        st.metric(
            "F1 Score",
            "81.33%"
        )

    st.markdown("---")

    st.subheader(
        "Current Severity Feature Importance"
    )

    current_importance = (
        prepare_importance_dataframe(
            feature_importance
        )
    )

    if current_importance is not None:

        current_importance = (
            current_importance
            .sort_values(
                "importance",
                ascending=False
            )
        )

        st.bar_chart(
            current_importance.set_index(
                "feature"
            )["importance"]
        )

    else:

        st.warning(
            "Could not find compatible feature/importance "
            "columns in feature_importance.csv."
        )

        st.write(
            "Columns found:",
            list(
                feature_importance.columns
            )
        )

    # ========================================================
    # FUTURE RISK MODEL
    # ========================================================

    st.markdown("---")

    st.header(
        "2. Future Fire Risk Model"
    )

    st.write(
        "Random Forest Classifier for next-day fire activity."
    )

    future_col1, future_col2, future_col3, future_col4 = (
        st.columns(4)
    )

    future_accuracy = (
        future_metadata.get(
            "accuracy",
            0
        )
    )

    future_precision = (
        future_metadata.get(
            "precision",
            0
        )
    )

    future_recall = (
        future_metadata.get(
            "recall",
            0
        )
    )

    future_f1 = (
        future_metadata.get(
            "f1",
            0
        )
    )

    with future_col1:

        st.metric(
            "Accuracy",
            f"{future_accuracy * 100:.2f}%"
        )

    with future_col2:

        st.metric(
            "Precision",
            f"{future_precision * 100:.2f}%"
        )

    with future_col3:

        st.metric(
            "Recall",
            f"{future_recall * 100:.2f}%"
        )

    with future_col4:

        st.metric(
            "F1 Score",
            f"{future_f1 * 100:.2f}%"
        )

    st.subheader(
        "Future Risk Feature Importance"
    )

    future_importance = (
        prepare_importance_dataframe(
            future_feature_importance
        )
    )

    if future_importance is not None:

        future_importance = (
            future_importance
            .sort_values(
                "importance",
                ascending=False
            )
        )

        st.bar_chart(
            future_importance.set_index(
                "feature"
            )["importance"]
        )

    else:

        st.warning(
            "Could not find compatible feature/importance "
            "columns in future_risk_feature_importance.csv."
        )

        st.write(
            "Columns found:",
            list(
                future_feature_importance.columns
            )
        )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Future Risk Model Information"
    )

    info_col1, info_col2 = (
        st.columns(2)
    )

    with info_col1:

        st.write(
            "**Grid Size:** "
            f"{future_metadata.get('grid_size', 0.25)}°"
        )

        st.write(
            "**Training/Test Split:** "
            f"{future_metadata.get('split_date', '2025-01-01')}"
        )

    with info_col2:

        st.write(
            "**Data Start:** "
            f"{future_metadata.get('date_start', 'N/A')}"
        )

        st.write(
            "**Data End:** "
            f"{future_metadata.get('date_end', 'N/A')}"
        )

    st.info(
        """
        The future-risk model uses a time-based split rather
        than a random split. Older observations are used for
        training and later observations are used for testing,
        which better represents a forecasting scenario.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Telangana Forest Fire Data Quality Enhancement "
    "& Predictive Analysis"
)

st.sidebar.caption(
    "Data source: NASA FIRMS"
)