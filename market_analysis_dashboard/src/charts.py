from __future__ import annotations
import plotly.graph_objects as go

def make_price_chart(data, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=data.index, y=data["ma_10"], mode="lines", name="MA 10"))
    fig.add_trace(go.Scatter(x=data.index, y=data["ma_20"], mode="lines", name="MA 20"))
    fig.update_layout(title=f"{ticker} Price and Moving Averages", height=520, xaxis_title="Date", yaxis_title="Price")
    return fig

def make_rsi_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["rsi_14"], mode="lines", name="RSI"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(title="RSI (14)", height=320)
    return fig

def make_macd_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["macd"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=data.index, y=data["macd_signal"], mode="lines", name="Signal"))
    fig.update_layout(title="MACD", height=320)
    return fig

def make_returns_histogram(data):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data["return_1d"].dropna(), nbinsx=40, name="Daily Returns"))
    fig.update_layout(title="Distribution of Daily Returns", height=320, xaxis_title="Return", yaxis_title="Frequency")
    return fig

def make_confusion_matrix_chart(cm):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Down", "Predicted Up"],
        y=["Actual Down", "Actual Up"],
        text=cm,
        texttemplate="%{text}",
        showscale=True
    ))
    fig.update_layout(title="Confusion Matrix", height=360)
    return fig

def make_actual_vs_predicted_returns(test_df, predicted_returns):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df["target_return"], mode="lines", name="Actual Return"))
    fig.add_trace(go.Scatter(x=test_df.index, y=predicted_returns, mode="lines", name="Predicted Return"))
    fig.update_layout(title="Actual vs Predicted Returns (Test Period)", height=360, xaxis_title="Date", yaxis_title="Return")
    return fig

def make_cumulative_return_chart(data):
    cumulative = (1 + data["return_1d"].fillna(0)).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=cumulative, mode="lines", name="Cumulative Growth"))
    fig.update_layout(title="Cumulative Growth of £1 Invested", height=320, yaxis_title="Growth")
    return fig
