# Market Predictive Modeling Dashboard

A generalized Streamlit app that applies the same predictive modeling workflow to **any stock, ETF, index, futures contract, or crypto ticker available through Yahoo Finance**.

## Supported examples
- Stocks: `AAPL`, `MSFT`, `TSLA`, `NVDA`
- ETFs: `SPY`, `QQQ`
- Indexes: `^GSPC`
- Futures: `ES=F`, `NQ=F`, `CL=F`, `GC=F`
- Crypto: `BTC-USD`, `ETH-USD`

## Features
- Fetches price data from Yahoo Finance with `yfinance`
- Engineers lagged returns, volatility, trend, and volume features
- Predicts next-period returns using multiple linear regression
- Uses a time-based train/test split
- Shows metrics, coefficients, charts, and auto-generated insights

## Run locally
```bash
pip install -r requirements.txt
streamlit run app_generalized.py
```

## Suggested repo name
`market-predictive-modeling-dashboard`
