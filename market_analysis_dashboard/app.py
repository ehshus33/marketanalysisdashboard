from __future__ import annotations
import streamlit as st
import pandas as pd



from src.data_utils import download_stock_data
from src.features import build_features, latest_indicator_summary
from src.models import (
    split_time_series,
    train_direction_model,
    evaluate_direction_model,
    train_return_model,
    evaluate_return_model,
    latest_prediction,
    baseline_accuracy,
)
from src.charts import (
    make_price_chart,
    make_rsi_chart,
    make_macd_chart,
    make_returns_histogram,
    make_confusion_matrix_chart,
    make_actual_vs_predicted_returns,
    make_cumulative_return_chart,
)

st.set_page_config(page_title="Market Analysis Dashboard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .section-note {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Market Analysis Dashboard")
st.caption("A cleaner and more complete stock analysis system for a final-year project presentation.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL")
    start_date = st.text_input("Start date", value="2018-01-01")
    direction_model_name = st.selectbox("Direction model", ["Logistic Regression", "Random Forest", "Neural Network"])
    return_model_name = st.selectbox("Return model", ["Linear Regression", "Neural Network"])
    test_size = st.slider("Test data size", 0.10, 0.40, 0.20, 0.05)
    run_button = st.button("Run analysis")

@st.cache_data(show_spinner=False)
def load_all_data(ticker: str, start_date: str):
    raw = download_stock_data(ticker, start_date)
    featured = build_features(raw)
    return raw, featured

if run_button:
    try:
        raw_df, model_df = load_all_data(ticker, start_date)
    except Exception as e:
        st.error(str(e))
        st.stop()

    train_df, test_df = split_time_series(model_df, test_size=test_size)

    direction_model = train_direction_model(train_df, direction_model_name)
    direction_metrics, direction_pred, direction_prob, cm = evaluate_direction_model(direction_model, test_df)

    return_model = train_return_model(train_df, return_model_name)
    return_metrics, return_pred = evaluate_return_model(return_model, test_df)

    full_direction_model = train_direction_model(model_df, direction_model_name)
    full_return_model = train_return_model(model_df, return_model_name)
    latest = latest_prediction(full_direction_model, full_return_model, model_df)

    indicator_summary = latest_indicator_summary(model_df)
    naive_acc = baseline_accuracy(test_df)

    overview_tab, indicators_tab, prediction_tab, evaluation_tab, advanced_tab, about_tab = st.tabs(
        ["Overview", "Indicators", "Predictions", "Evaluation", "Advanced", "About"]
    )

    with overview_tab:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted direction", latest["direction"])
        c2.metric("Probability of up day", f"{latest['prob_up'] * 100:.1f}%")
        c3.metric("Predicted next-day return", f"{latest['predicted_return'] * 100:.2f}%")
        c4.metric("Latest close", f"{model_df['close'].iloc[-1]:.2f}")

        st.plotly_chart(make_price_chart(model_df.tail(180), ticker.upper()), use_container_width=True)

        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(make_cumulative_return_chart(model_df), use_container_width=True)
        with right:
            st.markdown("### Quick interpretation")
            st.markdown(
                f"""
                <div class="section-note">
                <b>RSI status:</b> {indicator_summary['rsi_status']}<br>
                <b>Trend status:</b> {indicator_summary['trend_status']}<br>
                <b>MACD status:</b> {indicator_summary['macd_status']}<br><br>
                This turns the indicators into plain-English talking points for your presentation.
                </div>
                """,
                unsafe_allow_html=True
            )

    with indicators_tab:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(make_rsi_chart(model_df), use_container_width=True)
            st.plotly_chart(make_returns_histogram(model_df), use_container_width=True)
        with right:
            st.plotly_chart(make_macd_chart(model_df), use_container_width=True)
            st.dataframe(
                model_df[["close", "ma_10", "ma_20", "rsi_14", "macd", "macd_signal", "volatility_10"]].tail(10),
                use_container_width=True
            )

    with prediction_tab:
        st.subheader("Latest model output")
        p1, p2, p3 = st.columns(3)
        p1.metric("Direction output", latest["direction"])
        p2.metric("Probability of upward move", f"{latest['prob_up'] * 100:.1f}%")
        p3.metric("Expected return", f"{latest['predicted_return'] * 100:.2f}%")
        model_acc = direction_metrics["Accuracy"]
        comparison_df = pd.DataFrame({
    "Model": ["Baseline", direction_model_name],
    "Accuracy": [naive_acc, model_acc]
})

        st.dataframe(comparison_df, use_container_width=True)

        st.markdown(
            '<div class="section-note"><b>Presentation tip:</b> Say the model gives a probabilistic short-term signal, not a guaranteed future price.</div>',
            unsafe_allow_html=True
        )

        prediction_table = pd.DataFrame({
            "Actual Direction": test_df["target_up"].map({1: "Up", 0: "Down"}),
            "Predicted Direction": pd.Series(direction_pred, index=test_df.index).map({1: "Up", 0: "Down"}),
            "Probability Up": direction_prob,
            "Actual Return": test_df["target_return"],
            "Predicted Return": return_pred,
        }, index=test_df.index)

        st.dataframe(prediction_table.tail(20), use_container_width=True)

    with evaluation_tab:
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{direction_metrics['Accuracy'] * 100:.1f}%")
        m2.metric("F1 Score", f"{direction_metrics['F1 Score']:.2f}")
        m3.metric("Naive baseline accuracy", f"{naive_acc * 100:.1f}%")

        left, right = st.columns(2)
        with left:
            st.subheader("Classification metrics")
            st.write(direction_metrics)
            st.plotly_chart(make_confusion_matrix_chart(cm), use_container_width=True)

        with right:
            st.subheader("Regression metrics")
            st.write(return_metrics)
            st.plotly_chart(make_actual_vs_predicted_returns(test_df, return_pred), use_container_width=True)

        st.markdown(
            '<div class="section-note"><b>How to justify accuracy:</b> The model is trained on older data and tested on newer unseen data using a chronological split. That makes the evaluation more realistic for time-series forecasting.</div>',
            unsafe_allow_html=True
        )

    with advanced_tab:
        st.subheader("Advanced / experimental component")
        st.write({
            "Direction model used": direction_model_name,
            "Return model used": return_model_name
        })

        st.markdown(
            """
            If you choose **Neural Network**, this acts as the app's more advanced model.
            It is included as an experimental extension, while the simpler models remain easier to explain and justify.

            Good line for your presentation:

            *"I kept simpler baseline models at the core of the system for interpretability, and added a neural network as an advanced comparison model."*
            """
        )

    with about_tab:
        st.markdown(
            """
            ### Project summary

            **Aim:**  
            Build a stock market analysis dashboard that downloads historical stock data, calculates technical indicators, and predicts next-day market movement.

            **Main features:**  
            - Candlestick price chart
            - Moving averages
            - RSI and MACD
            - Daily returns analysis
            - Direction prediction
            - Return prediction
            - Evaluation metrics and confusion matrix

            **Why this design is better for a presentation:**  
            - More visually complete
            - Easier to explain than a full trading platform
            - Includes both core models and one advanced option
            - Shows clear evidence of evaluation

            **Limitations:**  
            The system only uses historical market data and technical indicators. It does not include news, sentiment, macroeconomic events, or company fundamentals.

            **Safe conclusion:**  
            The app does not promise perfect prediction. It aims to identify useful short-term patterns in historical data and evaluate them properly.
            """
        )
else:
    st.info("Set your options in the sidebar and click **Run analysis**.")
