import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Market Predictive Modeling Dashboard", layout="wide")

st.title("Market Predictive Modeling Dashboard")
st.markdown(
    "Run the same predictive modeling workflow on **any stock, ETF, index, or futures ticker available through Yahoo Finance (`yfinance`)**."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").strip().upper()

preset = st.sidebar.selectbox(
    "Quick examples",
    [
        "Custom",
        "AAPL (Apple)",
        "MSFT (Microsoft)",
        "TSLA (Tesla)",
        "NVDA (NVIDIA)",
        "SPY (S&P 500 ETF)",
        "QQQ (NASDAQ 100 ETF)",
        "^GSPC (S&P 500 Index)",
        "ES=F (E-mini S&P 500 Futures)",
        "NQ=F (Nasdaq Futures)",
        "CL=F (Crude Oil Futures)",
        "GC=F (Gold Futures)",
        "BTC-USD (Bitcoin)",
        "ETH-USD (Ethereum)"
    ]
)

preset_map = {
    "AAPL (Apple)": "AAPL",
    "MSFT (Microsoft)": "MSFT",
    "TSLA (Tesla)": "TSLA",
    "NVDA (NVIDIA)": "NVDA",
    "SPY (S&P 500 ETF)": "SPY",
    "QQQ (NASDAQ 100 ETF)": "QQQ",
    "^GSPC (S&P 500 Index)": "^GSPC",
    "ES=F (E-mini S&P 500 Futures)": "ES=F",
    "NQ=F (Nasdaq Futures)": "NQ=F",
    "CL=F (Crude Oil Futures)": "CL=F",
    "GC=F (Gold Futures)": "GC=F",
    "BTC-USD (Bitcoin)": "BTC-USD",
    "ETH-USD (Ethereum)": "ETH-USD",
}
if preset != "Custom":
    ticker = preset_map[preset]

interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

default_start = pd.to_datetime("2018-01-01")
default_end = pd.Timestamp.today().normalize()
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=default_end)

test_size = st.sidebar.slider("Test set proportion", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

selected_features = st.sidebar.multiselect(
    "Select model features",
    [
        "lag1_return", "lag2_return", "lag3_return",
        "range_pct", "oc_change_pct", "volatility_5",
        "volatility_10", "sma_gap_pct", "volume_change"
    ],
    default=[
        "lag1_return", "lag2_return", "lag3_return",
        "range_pct", "oc_change_pct", "volatility_5",
        "sma_gap_pct", "volume_change"
    ]
)

use_weekday = st.sidebar.checkbox("Include weekday dummies", value=False)
run_button = st.sidebar.button("Run analysis")

@st.cache_data(show_spinner=False)
def load_yahoo_data(ticker, start_date, end_date, interval):
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()

    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "Date"})

    return df

def engineer_features(raw_df):
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {missing}")

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df["return_1d"] = df["Close"].pct_change()
    df["lag1_return"] = df["return_1d"].shift(1)
    df["lag2_return"] = df["return_1d"].shift(2)
    df["lag3_return"] = df["return_1d"].shift(3)

    df["range_pct"] = np.where(df["Close"] != 0, (df["High"] - df["Low"]) / df["Close"], np.nan)
    df["oc_change_pct"] = np.where(df["Open"] != 0, (df["Close"] - df["Open"]) / df["Open"], np.nan)

    df["volatility_5"] = df["return_1d"].rolling(5).std()
    df["volatility_10"] = df["return_1d"].rolling(10).std()

    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_gap_pct"] = np.where(df["sma_10"] != 0, (df["sma_5"] - df["sma_10"]) / df["sma_10"], np.nan)

    df["volume_change"] = df["Volume"].pct_change()
    df["weekday"] = df["Date"].dt.day_name()
    df["target_next_return"] = df["Close"].pct_change().shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan)

    keep_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "lag1_return", "lag2_return", "lag3_return",
        "range_pct", "oc_change_pct", "volatility_5", "volatility_10",
        "sma_gap_pct", "volume_change", "weekday", "target_next_return"
    ]
    model_df = df[keep_cols].dropna().reset_index(drop=True)

    if model_df.empty:
        raise ValueError("No usable rows remain after feature engineering. Try a longer time range or a different interval.")

    return model_df

def prepare_xy(model_df, selected_features, use_weekday):
    X = model_df[selected_features].copy()

    if use_weekday:
        weekday_dummies = pd.get_dummies(model_df["weekday"], prefix="weekday", drop_first=True, dtype=float)
        X = pd.concat([X, weekday_dummies], axis=1)

    y = model_df["target_next_return"].copy()
    return X, y

def split_time_series(X, y, test_size):
    split_idx = int(len(X) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("Invalid split. Adjust date range or test size.")
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    return X_train, X_test, y_train, y_test

def generate_insights(ticker, coef_table, metrics):
    insights = []

    if metrics["R2_test"] < 0.02:
        insights.append(
            f"For {ticker}, short-horizon returns show very low linear predictability, which suggests noisy and near-efficient price behavior."
        )
    else:
        insights.append(
            f"For {ticker}, the model captures a small but non-zero amount of return variation, indicating weak but detectable linear structure."
        )

    if "sma_gap_pct" in coef_table.index:
        strength = "meaningful" if abs(coef_table["sma_gap_pct"]) > 0.05 else "modest"
        insights.append(
            f"Trend structure measured through SMA gap has a {strength} relationship with next-period returns."
        )

    if "volatility_5" in coef_table.index or "volatility_10" in coef_table.index:
        insights.append(
            "Volatility features help describe regime conditions, and prediction quality often deteriorates during more turbulent periods."
        )

    if any(col in coef_table.index for col in ["lag1_return", "lag2_return", "lag3_return"]):
        insights.append(
            "Lagged returns may reflect weak momentum or mean reversion, but their standalone predictive power is usually limited."
        )

    if "volume_change" in coef_table.index:
        insights.append(
            "Volume-change signals should be interpreted carefully, especially for assets or intervals where reported volume is sparse or irregular."
        )

    insights.append(
        "A practical next step is to compare this baseline with nonlinear models such as XGBoost, random forests, neural networks, or regime-aware strategies."
    )
    return insights

if run_button:
    try:
        if not ticker:
            st.error("Please enter a ticker symbol.")
            st.stop()

        with st.spinner(f"Downloading {ticker} from Yahoo Finance..."):
            raw_df = load_yahoo_data(ticker, str(start_date), str(end_date), interval)

        if raw_df.empty:
            st.error("No data returned. Check the ticker, interval, or date range.")
            st.stop()

        model_df = engineer_features(raw_df)

        if len(model_df) < 50:
            st.error("Not enough usable rows after feature engineering. Use a longer date range.")
            st.stop()

        X, y = prepare_xy(model_df, selected_features, use_weekday)
        X_train, X_test, y_train, y_test = split_time_series(X, y, test_size)

        model = LinearRegression()
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        metrics = {
            "MAE_train": mean_absolute_error(y_train, pred_train),
            "RMSE_train": np.sqrt(mean_squared_error(y_train, pred_train)),
            "R2_train": r2_score(y_train, pred_train),
            "MAE_test": mean_absolute_error(y_test, pred_test),
            "RMSE_test": np.sqrt(mean_squared_error(y_test, pred_test)),
            "R2_test": r2_score(y_test, pred_test),
        }

        coef_table = pd.Series(model.coef_, index=X.columns, name="coefficient").sort_values(
            key=lambda s: s.abs(), ascending=False
        )

        st.subheader("1) Asset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", ticker)
        c2.metric("Rows downloaded", len(raw_df))
        c3.metric("Rows modeled", len(model_df))
        c4.metric("Interval", interval)

        st.subheader("2) Raw Data Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.subheader("3) Engineered Modeling Dataset")
        st.dataframe(model_df.head(10), use_container_width=True)

        st.subheader("4) Model Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Train R²", f"{metrics['R2_train']:.6f}")
        m2.metric("Test R²", f"{metrics['R2_test']:.6f}")
        m3.metric("Intercept", f"{model.intercept_:.6f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("Train MAE", f"{metrics['MAE_train']:.6f}")
        m5.metric("Test MAE", f"{metrics['MAE_test']:.6f}")
        m6.metric("Test RMSE", f"{metrics['RMSE_test']:.6f}")

        st.subheader("5) Model Coefficients")
        st.dataframe(coef_table.to_frame(), use_container_width=True)

        st.subheader("6) Actual vs Predicted Returns")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(y_test.reset_index(drop=True), label="Actual")
        ax1.plot(pd.Series(pred_test), label="Predicted")
        ax1.set_xlabel("Test observations")
        ax1.set_ylabel("Target return")
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("7) Residual Distribution")
        residuals = y_test.reset_index(drop=True) - pd.Series(pred_test)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(residuals, bins=30)
        ax2.set_xlabel("Residual")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

        st.subheader("8) Price Context")
        plot_df = raw_df.copy().sort_values("Date").reset_index(drop=True)
        plot_df["SMA_5"] = plot_df["Close"].rolling(5).mean()
        plot_df["SMA_10"] = plot_df["Close"].rolling(10).mean()

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(plot_df["Date"], plot_df["Close"], label="Close")
        ax3.plot(plot_df["Date"], plot_df["SMA_5"], label="SMA 5")
        ax3.plot(plot_df["Date"], plot_df["SMA_10"], label="SMA 10")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Price")
        ax3.legend()
        st.pyplot(fig3)

        st.subheader("9) Portfolio-Ready Insights")
        for i, insight in enumerate(generate_insights(ticker, coef_table, metrics), 1):
            st.markdown(f"**{i}.** {insight}")

        st.subheader("10) Download Engineered Dataset")
        csv_bytes = model_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download engineered dataset as CSV",
            data=csv_bytes,
            file_name=f"{ticker.replace('^', '').replace('=', '_')}_model_dataset.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Choose a ticker and click **Run analysis**.")
    st.markdown(
        '''
### Supported ideas
- Stocks: `AAPL`, `MSFT`, `TSLA`
- ETFs: `SPY`, `QQQ`
- Indexes: `^GSPC`, `^IXIC`
- Futures: `ES=F`, `NQ=F`, `CL=F`, `GC=F`
- Crypto: `BTC-USD`, `ETH-USD`

### What the app does
- Downloads market data from Yahoo Finance
- Engineers the same features across different assets
- Runs the same regression workflow
- Produces metrics, charts, coefficients, and insights
        '''
    )
